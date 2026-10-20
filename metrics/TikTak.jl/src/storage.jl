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
    screening::Bool             # evaluated by the screening objective of a staged run
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
    SavedEvaluation

A snapshot of one cached parameter point returned by [`load_evaluations`](@ref)
or retained in `result.evaluations` with `store_evaluations=true`.

`parameters` are in original model units; `unit` contains the free unit-box
coordinates. `value` is the loss (`NaN` unless `status == :ok`). `moments` and
`residuals` are vectors when supplied by the objective, otherwise `nothing`.
`status` is `:ok`, `:failed`, `:error`, `:abandoned`, or `:pending`; `error`
contains the failure message when available. `seconds` is evaluation runtime.

`screening` is true only for the cheaper objective in a staged run, whose
values must not be compared with full-accuracy values. `key` identifies the
point and stage; `sequence` orders model-call attempts. Cached repeats appear
once, and a retried abandoned point carries its latest attempt. Thus the number
of records can be smaller than `result.n_evals`. Vectors are independent copies
and can be edited without changing the optimizer or saved journal.
"""
struct SavedEvaluation
    key::String
    sequence::Int
    screening::Bool
    unit::Vector{Float64}
    parameters::Vector{Float64}
    status::Symbol
    value::Float64
    moments::Union{Nothing,Vector{Float64}}
    residuals::Union{Nothing,Vector{Float64}}
    error::Union{Nothing,String}
    seconds::Float64
end

function _saved_evaluation(r)
    return SavedEvaluation(r.key, r.sequence, r.screening, copy(r.unit),
                           copy(r.parameters), r.status, r.value,
                           r.moments === nothing ? nothing : copy(r.moments),
                           r.residuals === nothing ? nothing : copy(r.residuals),
                           r.error, r.seconds)
end

function _evaluation_snapshot(records; status::Symbol=:all, stage::Symbol=:all)
    status in (:all, :ok, :failed, :error, :abandoned, :pending) ||
        throw(ArgumentError("status must be :all, :ok, :failed, :error, :abandoned, or :pending"))
    stage in (:all, :full, :screening) ||
        throw(ArgumentError("stage must be :all, :full, or :screening"))
    rows = SavedEvaluation[
        _saved_evaluation(r) for r in records
        if (status === :all || r.status === status) &&
           (stage === :all || r.screening == (stage === :screening))
    ]
    sort!(rows; by=r -> r.sequence)
    return rows
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
    return Dict{String,Any}(
        "index" => r.index,
        "unit" => r.unit,
        "x" => r.x,
        "fun" => r.fun,
        "converged" => r.converged,
        "message" => r.message,
    )
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

# Screening evaluations of a staged run come from a different (cheaper) objective,
# so they are cached under their own key and never answer a full-accuracy lookup.
function point_key(unit; screening::Bool=false)
    bytes = collect(reinterpret(UInt8, _canonical(unit)))
    screening && append!(bytes, codeunits("screen"))
    return bytes2hex(sha256(bytes))
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

function _apply!(store::Store, record::Dict{String,Any})
    record_type = record["t"]
    if record_type == "meta"
        store.meta[record["k"]] = record["v"]
    elseif record_type == "claim"
        key, task = record["key"], record["task"]
        store.n_attempts += 1
        store.evaluations[key] = EvalRecord(
            key,
            store.n_attempts,
            get(record, "screening", false),
            _float_vector(record["unit"]),
            _float_vector(record["parameters"]),
            :pending,
            NaN,
            nothing,
            nothing,
            nothing,
            0.0,
            Float64(record["started"]),
        )
        store.attempts_by_task[task] = get(store.attempts_by_task, task, 0) + 1
        push!(get!(store.task_points, task, Set{String}()), key)
    elseif record_type == "hit"
        push!(get!(store.task_points, record["task"], Set{String}()), record["key"])
    elseif record_type == "finish"
        entry = store.evaluations[record["key"]]
        entry.status = Symbol(record["status"])
        entry.value = record["value"] === nothing ? NaN : Float64(record["value"])
        entry.moments = _float_vector(record["moments"])
        entry.residuals = _float_vector(record["residuals"])
        entry.error = record["error"]
        entry.seconds = Float64(record["seconds"])
        entry.updated = Float64(record["updated"])
        # Screening values of a staged run are not comparable with full-accuracy ones.
        if entry.status === :ok && !entry.screening
            incumbent = store.best_key === nothing ? nothing : store.evaluations[store.best_key]
            if incumbent === nothing ||
               (entry.value, entry.sequence, entry.key) < (incumbent.value, incumbent.sequence, incumbent.key)
                store.best_key = entry.key
            end
        end
    elseif record_type == "abandon"
        entry = get(store.evaluations, record["key"], nothing)
        entry !== nothing && entry.status === :pending && (entry.status = :abandoned)
    elseif record_type == "abandon_all"
        for entry in values(store.evaluations)
            entry.status === :pending && (entry.status = :abandoned)
        end
    elseif record_type == "local"
        id = Int(record["id"])
        store.locals[id] = LocalRecord(
            id, _float_vector(record["start"]), _float_vector(record["seed"]), :pending, nothing,
        )
    elseif record_type == "local_done"
        entry = store.locals[Int(record["id"])]
        entry.status = Symbol(record["status"])
        entry.result = record["result"] === nothing ? nothing : LocalResult(record["result"])
    else
        throw(ErrorException("unknown journal record type $(repr(record_type))"))
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
    claim!(store, unit, parameters, task, local_limit; screening=false) -> (status, key, payload)

Reserve a budget slot for a new model call, or report a cached outcome. `status`
is one of `:claimed` (evaluate now), `:ok`/`:failed` (cached; `payload` is the
value for `:ok`), `:error` (cached unexpected error; `payload` is its message),
`:pending` (another worker is evaluating this exact point; retry later),
`:budget`, or `:local_budget`. `screening=true` marks an evaluation by the
screening objective of a staged run, which has its own cache entry.
"""
function claim!(store::Store, unit, parameters, task::AbstractString, local_limit; screening::Bool=false)
    unit = _canonical(unit)
    key = point_key(unit; screening=screening)
    lock(store.lock) do
        cached_record = get(store.evaluations, key, nothing)
        if cached_record !== nothing && cached_record.status in (:ok, :failed, :error)
            if !(key in get(store.task_points, task, Set{String}()))
                _commit!(store, Dict{String,Any}("t" => "hit", "task" => task, "key" => key))
            end
            cached_record.status === :error && return (:error, key, something(cached_record.error, ""))
            return (cached_record.status, key, cached_record.status === :ok ? cached_record.value : nothing)
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
        if cached_record === nothing || cached_record.status === :abandoned
            claim_record = Dict{String,Any}(
                "t" => "claim",
                "key" => key,
                "unit" => unit,
                "parameters" => Vector{Float64}(parameters),
                "task" => String(task),
                "started" => time(),
            )
            screening && (claim_record["screening"] = true)
            _commit!(store, claim_record)
            return (:claimed, key, nothing)
        end
        return (:pending, key, nothing)
    end
end

function finish!(store::Store, key::AbstractString; value=nothing, moments=nothing, residuals=nothing,
                 error=nothing, seconds=0.0, unexpected::Bool=false)
    status = unexpected ? "error" : (value === nothing ? "failed" : "ok")
    lock(store.lock) do
        _commit!(store, Dict{String,Any}(
            "t" => "finish",
            "key" => String(key),
            "status" => status,
            "value" => value,
            "moments" => moments,
            "residuals" => residuals,
            "error" => error,
            "seconds" => Float64(seconds),
            "updated" => time(),
        ))
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

Best finite full-accuracy evaluation overall, or among the points touched by
`task`; screening evaluations of a staged run are excluded. Ties are broken in
favour of the point attempted first, then by key, so the choice is
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
            record = store.evaluations[key]
            (record.status === :ok && !record.screening) || continue
            if winner === nothing || (record.value, record.sequence, record.key) < (winner.value, winner.sequence, winner.key)
                winner = record
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
        _commit!(store, Dict{String,Any}(
            "t" => "local",
            "id" => Int(index),
            "start" => Vector{Float64}(start),
            "seed" => Vector{Float64}(seed),
        ); sync=true)
    end
    return nothing
end

function finish_local!(store::Store, index::Integer, result::LocalResult, complete::Bool)
    lock(store.lock) do
        _commit!(store, Dict{String,Any}(
            "t" => "local_done",
            "id" => Int(index),
            "status" => complete ? "done" : "pending",
            "result" => Dict(result),
        ); sync=true)
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
reevaluation. In a staged run, full-accuracy evaluations are ranked first and
screening evaluations fill any remaining slots.
"""
function load_estimates(directory::AbstractString; limit::Integer=20)
    limit > 0 || throw(ArgumentError("limit must be a positive integer"))
    isfile(joinpath(directory, JOURNAL_NAME)) || throw(ArgumentError("no run history found in $directory"))
    store = Store(directory; readonly=true)
    records = [r for r in values(store.evaluations) if r.status === :ok]
    sort!(records; by=r -> (r.screening, r.value, r.sequence, r.key))
    return [copy(r.parameters) for r in records[1:min(limit, end)]]
end

"""
    load_evaluations(directory; status=:all, stage=:all) -> Vector{SavedEvaluation}
    load_evaluations(result::TikTakResult; status=:all, stage=:all)

Retrieve paired parameter points, losses, moments, residuals, and diagnostics
in model-call attempt order, without evaluating the model. Pass a saved run
directory or a result. A result uses its retained workspace snapshot if present,
otherwise its `run_dir`. A retained snapshot describes that invocation; load
the directory to read subsequent resumes.

By default all cached points and statuses are included. Use `status=:ok` for
successful evaluations and `stage=:full` to exclude cheaper screening values
from a staged run (`stage=:screening` selects only those cheaper evaluations).
Other status filters are `:failed`, `:error`, `:abandoned`, and `:pending`.
Each call returns independent copies; see [`SavedEvaluation`](@ref).

For nonempty successful records with equally sized moment vectors:
```julia
rows = load_evaluations(result; status=:ok, stage=:full)
parameters = stack([r.parameters for r in rows]; dims=1) # evaluations × parameters
moments = stack([r.moments for r in rows]; dims=1)       # evaluations × moments
losses = [r.value for r in rows]
```
Scalar objectives have `moments === nothing` unless they return an `Evaluation`
containing moments. Failed evaluations may also lack moments.
"""
function load_evaluations(directory::AbstractString; status::Symbol=:all, stage::Symbol=:all)
    isfile(joinpath(directory, JOURNAL_NAME)) ||
        throw(ArgumentError("no run history found in $directory"))
    return _evaluation_snapshot(Store(directory; readonly=true); status=status, stage=stage)
end

function _evaluation_snapshot(store::Store; kwargs...)
    lock(store.lock) do
        return _evaluation_snapshot(values(store.evaluations); kwargs...)
    end
end
