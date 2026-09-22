# Coordinator-owned evaluation cache, budgets, and restart state.
#
# The coordinator process holds the run state in memory and appends every state
# change to `history.jsonl` in the run directory before applying it. Workers
# never touch the disk: they reach the store through `RemoteStore`, which turns
# every call into a `remotecall_fetch` to the coordinator (see local.jl). Because
# the journal is replayed on open, a run can be resumed after an interruption
# with every completed evaluation and every budget charge intact.

"""The global evaluation or wall-clock budget was reached."""
struct BudgetExhausted <: Exception end

"""The local search used its evaluation allocation."""
struct LocalBudgetExhausted <: Exception end

const JOURNAL_NAME = "history.jsonl"
const RESULT_NAME = "result.json"
const LOCK_NAME = "coordinator.lock"

mutable struct EvalRecord
    key::String
    sequence::Int               # attempt order; breaks value ties deterministically
    unit::Vector{Float64}
    parameters::Vector{Float64}
    status::Symbol              # :pending, :ok, :failed, :error, :abandoned
    value::Float64              # NaN unless status == :ok
    moments::Union{Nothing,Vector{Float64}}
    residuals::Union{Nothing,Vector{Float64}}
    error::Union{Nothing,String}
    seconds::Float64
    updated::Float64
end

"""
    LocalResult

Outcome of one local search: its seed `index`, the best point it evaluated in
unit (`unit`) and physical (`x`) coordinates with objective `fun` (all
`nothing` if it never found a finite point), whether the local solver reported
convergence, and the solver's message.
"""
struct LocalResult
    index::Int
    unit::Union{Nothing,Vector{Float64}}
    x::Union{Nothing,Vector{Float64}}
    fun::Union{Nothing,Float64}
    converged::Bool
    message::String
end

function Base.Dict(r::LocalResult)
    return Dict{String,Any}("index" => r.index, "unit" => r.unit, "x" => r.x, "fun" => r.fun,
                            "converged" => r.converged, "message" => r.message)
end

function LocalResult(d::AbstractDict)
    return LocalResult(Int(d["index"]), _float_vector(d["unit"]), _float_vector(d["x"]),
                       d["fun"] === nothing ? nothing : Float64(d["fun"]),
                       Bool(d["converged"]), String(d["message"]))
end

mutable struct LocalRecord
    id::Int
    start::Vector{Float64}
    seed::Vector{Float64}
    status::Symbol              # :pending or :done
    result::Union{Nothing,LocalResult}
end

mutable struct Store
    directory::String
    io::Union{Nothing,IOStream}
    lock::ReentrantLock
    meta::Dict{String,Any}
    evaluations::Dict{String,EvalRecord}
    n_attempts::Int
    attempts_by_task::Dict{String,Int}
    task_points::Dict{String,Set{String}}
    locals::Dict{Int,LocalRecord}
    best_key::Union{Nothing,String}
end

_float_vector(::Nothing) = nothing
_float_vector(v) = Vector{Float64}(v)

# Canonicalize signed zero so that -0.0 and 0.0 share a key; never round distinct points.
_canonical(unit) = Vector{Float64}(unit) .+ 0.0

function point_key(unit)
    u = _canonical(unit)
    return bytes2hex(sha256(collect(reinterpret(UInt8, u))))
end

"""
    Store(directory; readonly=false)

Open (and replay) the run journal in `directory`. With `readonly=true` no file
is created or appended, which is how [`load_estimates`](@ref) reads a run.
"""
function Store(directory::AbstractString; readonly::Bool=false)
    directory = abspath(String(directory))
    readonly || mkpath(directory)
    store = Store(directory, nothing, ReentrantLock(), Dict{String,Any}(), Dict{String,EvalRecord}(),
                  0, Dict{String,Int}(), Dict{String,Set{String}}(), Dict{Int,LocalRecord}(), nothing)
    path = joinpath(directory, JOURNAL_NAME)
    isfile(path) && _replay!(store, path)
    readonly || (store.io = open(path, "a"))
    return store
end

function Base.close(store::Store)
    lock(store.lock) do
        if store.io !== nothing
            _fsync(store.io)
            close(store.io)
            store.io = nothing
        end
    end
    return nothing
end

function _fsync(io::IOStream)
    flush(io)
    Sys.iswindows() || ccall(:fsync, Cint, (RawFD,), fd(io))
    return nothing
end

function _replay!(store::Store, path)
    lines = readlines(path)
    for (i, line) in enumerate(lines)
        isempty(strip(line)) && continue
        record = try
            JSON.parse(line; dicttype=Dict{String,Any})
        catch err
            # A crash can leave a partial final line; anything else is corruption.
            i == length(lines) && continue
            throw(ErrorException("corrupt journal line $i in $path: $(sprint(showerror, err))"))
        end
        _apply!(store, record)
    end
    return store
end

# Every state change goes through a journal record so that replay reproduces
# the in-memory state exactly. `sync` forces the record to stable storage.
function _commit!(store::Store, record::Dict{String,Any}; sync::Bool=false)
    if store.io !== nothing
        JSON.json(store.io, record)
        write(store.io, '\n')
        sync ? _fsync(store.io) : flush(store.io)
    end
    _apply!(store, record)
    return nothing
end

function _apply!(store::Store, r::Dict{String,Any})
    t = r["t"]
    if t == "meta"
        store.meta[r["k"]] = r["v"]
    elseif t == "claim"
        key, task = r["key"], r["task"]
        store.n_attempts += 1
        store.evaluations[key] = EvalRecord(key, store.n_attempts, _float_vector(r["unit"]),
                                            _float_vector(r["parameters"]), :pending, NaN, nothing, nothing,
                                            nothing, 0.0, Float64(r["started"]))
        store.attempts_by_task[task] = get(store.attempts_by_task, task, 0) + 1
        push!(get!(store.task_points, task, Set{String}()), key)
    elseif t == "hit"
        push!(get!(store.task_points, r["task"], Set{String}()), r["key"])
    elseif t == "finish"
        rec = store.evaluations[r["key"]]
        rec.status = Symbol(r["status"])
        rec.value = r["value"] === nothing ? NaN : Float64(r["value"])
        rec.moments = _float_vector(r["moments"])
        rec.residuals = _float_vector(r["residuals"])
        rec.error = r["error"]
        rec.seconds = Float64(r["seconds"])
        rec.updated = Float64(r["updated"])
        if rec.status === :ok
            best = store.best_key === nothing ? nothing : store.evaluations[store.best_key]
            if best === nothing || (rec.value, rec.sequence, rec.key) < (best.value, best.sequence, best.key)
                store.best_key = rec.key
            end
        end
    elseif t == "abandon"
        rec = get(store.evaluations, r["key"], nothing)
        rec !== nothing && rec.status === :pending && (rec.status = :abandoned)
    elseif t == "abandon_all"
        for rec in values(store.evaluations)
            rec.status === :pending && (rec.status = :abandoned)
        end
    elseif t == "local"
        id = Int(r["id"])
        store.locals[id] = LocalRecord(id, _float_vector(r["start"]), _float_vector(r["seed"]), :pending, nothing)
    elseif t == "local_done"
        rec = store.locals[Int(r["id"])]
        rec.status = Symbol(r["status"])
        rec.result = r["result"] === nothing ? nothing : LocalResult(r["result"])
    else
        throw(ErrorException("unknown journal record type $(repr(t))"))
    end
    return nothing
end

# Structural comparison of two specifications after a JSON round trip.
_normalize(x::AbstractDict) = Dict{String,Any}(String(k) => _normalize(v) for (k, v) in x)
_normalize(x::AbstractVector) = Any[_normalize(v) for v in x]
_normalize(x::Tuple) = Any[_normalize(v) for v in x]
_normalize(x) = x

function initialize!(store::Store, specification; resume::Bool, max_evals::Integer, deadline)
    lock(store.lock) do
        existing = get(store.meta, "specification", nothing)
        if existing !== nothing
            resume || throw(ArgumentError(
                "run already exists in $(store.directory); choose a new directory or resume=true"))
            _normalize(existing) == _normalize(specification) || throw(ArgumentError(
                "restart specification differs from the saved run; use a new run directory and warm_start"))
        elseif resume
            throw(ArgumentError("there is no initialized run to resume in $(store.directory)"))
        end
        put!(store, "specification", specification)
        put!(store, "max_evals", Int(max_evals))
        put!(store, "deadline", deadline)
        # Keep charges for interrupted calls: work may have been performed.
        _commit!(store, Dict{String,Any}("t" => "abandon_all"); sync=true)
    end
    return nothing
end

function Base.get(store::Store, key::AbstractString, default=nothing)
    lock(store.lock) do
        return get(store.meta, key, default)
    end
end

function Base.put!(store::Store, key::AbstractString, value)
    lock(store.lock) do
        _commit!(store, Dict{String,Any}("t" => "meta", "k" => key, "v" => value); sync=true)
    end
    return nothing
end

n_attempts(store::Store) = lock(() -> store.n_attempts, store.lock)

function exhausted(store::Store)
    lock(store.lock) do
        deadline = get(store.meta, "deadline", nothing)
        return store.n_attempts >= store.meta["max_evals"] ||
               (deadline !== nothing && time() >= deadline)
    end
end

"""
    claim!(store, unit, parameters, task, local_limit) -> (status, key, payload)

Reserve a budget slot for a new model call, or report a cached outcome. `status`
is one of `:claimed` (evaluate now), `:ok`/`:failed` (cached; `payload` is the
value for `:ok`), `:error` (cached unexpected error; `payload` is its message),
`:pending` (another worker is evaluating this exact point; retry later),
`:budget`, or `:local_budget`.
"""
function claim!(store::Store, unit, parameters, task::AbstractString, local_limit)
    unit = _canonical(unit)
    key = point_key(unit)
    lock(store.lock) do
        rec = get(store.evaluations, key, nothing)
        if rec !== nothing && rec.status in (:ok, :failed, :error)
            if !(key in get(store.task_points, task, Set{String}()))
                _commit!(store, Dict{String,Any}("t" => "hit", "task" => task, "key" => key))
            end
            rec.status === :error && return (:error, key, something(rec.error, ""))
            return (rec.status, key, rec.status === :ok ? rec.value : nothing)
        end
        if store.n_attempts >= store.meta["max_evals"]
            return (:budget, key, nothing)
        end
        deadline = get(store.meta, "deadline", nothing)
        if deadline !== nothing && time() >= deadline
            return (:budget, key, nothing)
        end
        if local_limit !== nothing && get(store.attempts_by_task, task, 0) >= local_limit
            return (:local_budget, key, nothing)
        end
        if rec === nothing || rec.status === :abandoned
            _commit!(store, Dict{String,Any}("t" => "claim", "key" => key, "unit" => unit,
                                             "parameters" => Vector{Float64}(parameters),
                                             "task" => String(task), "started" => time()))
            return (:claimed, key, nothing)
        end
        return (:pending, key, nothing)
    end
end

function finish!(store::Store, key::AbstractString; value=nothing, moments=nothing, residuals=nothing,
                 error=nothing, seconds=0.0, unexpected::Bool=false)
    status = unexpected ? "error" : (value === nothing ? "failed" : "ok")
    lock(store.lock) do
        _commit!(store, Dict{String,Any}("t" => "finish", "key" => String(key), "status" => status,
                                         "value" => value, "moments" => moments, "residuals" => residuals,
                                         "error" => error, "seconds" => Float64(seconds), "updated" => time()))
    end
    return nothing
end

function abandon!(store::Store, key::AbstractString)
    lock(store.lock) do
        _commit!(store, Dict{String,Any}("t" => "abandon", "key" => String(key)); sync=true)
    end
    return nothing
end

"""
    best(store, task=nothing) -> Union{Nothing, EvalRecord}

Best finite evaluation overall, or among the points touched by `task`. Ties are
broken in favour of the point attempted first, then by key, so the choice is
deterministic and stable under replay.
"""
function best(store::Store, task=nothing)
    lock(store.lock) do
        if task === nothing
            return store.best_key === nothing ? nothing : store.evaluations[store.best_key]
        end
        keys = get(store.task_points, task, nothing)
        keys === nothing && return nothing
        winner = nothing
        for key in keys
            rec = store.evaluations[key]
            rec.status === :ok || continue
            if winner === nothing || (rec.value, rec.sequence, rec.key) < (winner.value, winner.sequence, winner.key)
                winner = rec
            end
        end
        return winner
    end
end

function n_failed(store::Store)
    lock(store.lock) do
        return count(r -> r.status in (:failed, :error, :abandoned), values(store.evaluations))
    end
end

function local_rows(store::Store)
    lock(store.lock) do
        return [store.locals[id] for id in sort!(collect(keys(store.locals)))]
    end
end

local_row(store::Store, index::Integer) = lock(() -> store.locals[Int(index)], store.lock)

function create_local!(store::Store, index::Integer, start, seed)
    lock(store.lock) do
        _commit!(store, Dict{String,Any}("t" => "local", "id" => Int(index), "start" => Vector{Float64}(start),
                                         "seed" => Vector{Float64}(seed)); sync=true)
    end
    return nothing
end

function finish_local!(store::Store, index::Integer, result::LocalResult, complete::Bool)
    lock(store.lock) do
        _commit!(store, Dict{String,Any}("t" => "local_done", "id" => Int(index),
                                         "status" => complete ? "done" : "pending",
                                         "result" => Dict(result)); sync=true)
    end
    return nothing
end

# Only the coordinator exports the human-readable snapshot; the rename is atomic.
function export_result(store::Store, result::AbstractDict)
    temporary = joinpath(store.directory, RESULT_NAME * ".tmp")
    open(temporary, "w") do io
        JSON.json(io, result; pretty=true)
        write(io, '\n')
        _fsync(io)
    end
    mv(temporary, joinpath(store.directory, RESULT_NAME); force=true)
    return nothing
end

"""
    coordinator_lock(directory) -> lock

Advisory lock that prevents two coordinators from owning the same run directory.
`close(lock)` releases it; a stale lock left by a dead process is reclaimed.
"""
function coordinator_lock(directory::AbstractString)
    mkpath(directory)
    try
        return mkpidlock(joinpath(directory, LOCK_NAME); wait=false)
    catch err
        err isa PidlockedError || rethrow()
        throw(ErrorException("another coordinator is using this run directory: $directory"))
    end
end

function coordinator_lock(f::Function, directory::AbstractString)
    handle = coordinator_lock(directory)
    try
        return f()
    finally
        close(handle)
    end
end

"""
    load_estimates(directory; limit=20) -> Vector{Vector{Float64}}

Read the best evaluated physical parameter vectors of a saved run, for use as
`warm_start` in a NEW search. Their objective values are deliberately not
imported: a changed model, weighting matrix, or simulation design requires
reevaluation.
"""
function load_estimates(directory::AbstractString; limit::Integer=20)
    limit > 0 || throw(ArgumentError("limit must be a positive integer"))
    isfile(joinpath(directory, JOURNAL_NAME)) || throw(ArgumentError("no run history found in $directory"))
    store = Store(directory; readonly=true)
    records = [r for r in values(store.evaluations) if r.status === :ok]
    sort!(records; by=r -> (r.value, r.sequence, r.key))
    return [copy(r.parameters) for r in records[1:min(limit, end)]]
end
