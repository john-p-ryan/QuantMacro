# Sobol screening and asynchronous TikTak multistart coordination.

_positive_int(x) = x isa Integer && !(x isa Bool) && x >= 1

"""
    TikTakConfig(; kwargs...)
    TikTakConfig(base::TikTakConfig; kwargs...)

Algorithm settings. The second form copies `base` with overrides.

- `n_samples=256`: scrambled Sobol screening points (powers of two preserve balance).
- `n_local=nothing`: retained seeds; default 10% of `n_samples`, rounded up.
- `max_evals=10_000`: hard cumulative cap on model-call attempts, including
  failed and interrupted ones, across resumes.
- `local_max_evals=200`: new model attempts allowed per local search.
- `seed=0`: seed of the Sobol scrambling.
- `local_method=NelderMeadLocal()`: a [`LocalMethod`](@ref).
- `initial_step=0.1`, `x_tol=1e-5`: initial step and final resolution of the
  local search, in unit-box coordinates; require `0 < x_tol < initial_step <= 0.5`.
- `f_tol=1e-8`: absolute function tolerance (improvement threshold for pattern search).
- `mixing_min=0.1`, `mixing_max=0.995`, `mixing_power=0.5`: the weight on the
  incumbent when starting seed `i` of `n` is `clamp((i/n)^mixing_power, mixing_min, mixing_max)`.
- `max_seconds=nothing`: soft per-invocation deadline checked before new model calls.
- `target_value=nothing`: stop once an evaluation reaches this objective value.
- `failure_exceptions=(ModelEvaluationError,)`: exception types recorded as
  expected failures rather than propagated as errors.

Only `max_evals` and `max_seconds` may change when a run is resumed.
"""
struct TikTakConfig
    n_samples::Int
    n_local::Union{Nothing,Int}
    max_evals::Int
    local_max_evals::Int
    seed::Int
    local_method::LocalMethod
    initial_step::Float64
    x_tol::Float64
    f_tol::Float64
    mixing_min::Float64
    mixing_max::Float64
    mixing_power::Float64
    max_seconds::Union{Nothing,Float64}
    target_value::Union{Nothing,Float64}
    failure_exceptions::Tuple

    function TikTakConfig(; n_samples=256, n_local=nothing, max_evals=10_000, local_max_evals=200, seed=0,
                          local_method=NelderMeadLocal(), initial_step=0.1, x_tol=1e-5, f_tol=1e-8,
                          mixing_min=0.1, mixing_max=0.995, mixing_power=0.5, max_seconds=nothing,
                          target_value=nothing, failure_exceptions=(ModelEvaluationError,))
        for (name, value) in (("n_samples", n_samples), ("max_evals", max_evals), ("local_max_evals", local_max_evals))
            _positive_int(value) || throw(ArgumentError("$name must be a positive integer"))
        end
        (n_local === nothing || _positive_int(n_local)) ||
            throw(ArgumentError("n_local must be a positive integer or nothing"))
        (seed isa Integer && !(seed isa Bool) && seed >= 0) ||
            throw(ArgumentError("seed must be a nonnegative integer"))
        local_method isa LocalMethod || throw(ArgumentError("local_method must be a LocalMethod"))
        (0 < x_tol < initial_step <= 0.5) ||
            throw(ArgumentError("require 0 < x_tol < initial_step <= 0.5 in unit coordinates"))
        (isfinite(f_tol) && f_tol >= 0) || throw(ArgumentError("f_tol must be finite and nonnegative"))
        (0 <= mixing_min <= mixing_max < 1) || throw(ArgumentError("require 0 <= mixing_min <= mixing_max < 1"))
        (isfinite(mixing_power) && mixing_power > 0) || throw(ArgumentError("mixing_power must be positive and finite"))
        (max_seconds === nothing || (isfinite(max_seconds) && max_seconds > 0)) ||
            throw(ArgumentError("max_seconds must be positive and finite"))
        (target_value === nothing || isfinite(target_value)) || throw(ArgumentError("target_value must be finite"))
        failures = Tuple(failure_exceptions)
        (all(T -> T isa Type && T <: Exception, failures) && ModelEvaluationError in failures) ||
            throw(ArgumentError("failure_exceptions must be exception types and include ModelEvaluationError"))
        return new(Int(n_samples), n_local === nothing ? nothing : Int(n_local), Int(max_evals),
                   Int(local_max_evals), Int(seed), local_method, Float64(initial_step), Float64(x_tol),
                   Float64(f_tol), Float64(mixing_min), Float64(mixing_max), Float64(mixing_power),
                   max_seconds === nothing ? nothing : Float64(max_seconds),
                   target_value === nothing ? nothing : Float64(target_value), failures)
    end
end

function TikTakConfig(base::TikTakConfig; kwargs...)
    fields = Dict{Symbol,Any}(name => getfield(base, name) for name in fieldnames(TikTakConfig))
    merge!(fields, Dict{Symbol,Any}(kwargs))
    return TikTakConfig(; fields...)
end

_type_name(T::Type) = string(parentmodule(T), ".", nameof(T))

function specification(config::TikTakConfig)
    spec = Dict{String,Any}()
    for name in fieldnames(TikTakConfig)
        # Budgets can change on resume; algorithm settings cannot.
        name in (:max_evals, :max_seconds) && continue
        spec[string(name)] = getfield(config, name)
    end
    spec["local_method"] = specification(config.local_method)
    spec["failure_exceptions"] = [_type_name(T) for T in config.failure_exceptions]
    return spec
end

"""
    TikTakResult

Outcome of [`minimize`](@ref): best physical parameters `x` with objective `fun`
(and `moments`/`residuals` when the objective supplied them), the number of
model-call attempts `n_evals`, the number of distinct failed/error/abandoned
points `n_failed`, the number of completed local searches, a `status`
(`:completed`, `:target_reached`, `:budget_exhausted`, `:no_feasible_point`, or
`:running` in callbacks), a `message`, the `run_dir`, and the
[`LocalResult`](@ref)s. `status == :completed` means the allocated searches
finished, not that every local solver converged; [`has_solution`](@ref) means a
finite feasible estimate exists, not global optimality.
"""
struct TikTakResult
    x::Union{Nothing,Vector{Float64}}
    fun::Float64
    moments::Union{Nothing,Vector{Float64}}
    residuals::Union{Nothing,Vector{Float64}}
    n_evals::Int
    n_failed::Int
    n_local_completed::Int
    status::Symbol
    message::String
    run_dir::String
    local_results::Vector{LocalResult}
end

"""A feasible estimate was found; this is not a global-optimality certificate."""
has_solution(r::TikTakResult) = r.x !== nothing && isfinite(r.fun)

function to_dict(r::TikTakResult)
    return Dict{String,Any}(
        "x" => r.x, "fun" => isfinite(r.fun) ? r.fun : nothing, "moments" => r.moments,
        "residuals" => r.residuals, "n_evals" => r.n_evals, "n_failed" => r.n_failed,
        "n_local_completed" => r.n_local_completed, "status" => string(r.status),
        "message" => r.message, "run_dir" => r.run_dir,
        "local_results" => [Dict(l) for l in r.local_results])
end

function Base.show(io::IO, r::TikTakResult)
    print(io, "TikTakResult(status=:", r.status, ", fun=", r.fun, ", x=", r.x,
          ", n_evals=", r.n_evals, ", n_failed=", r.n_failed,
          ", n_local_completed=", r.n_local_completed, ")")
end

"""
    sobol_points(dimension, n, seed) -> Vector{Vector{Float64}}

The first `n` points of a digitally scrambled Sobol sequence in `[0, 1)^dimension`.
The full next power-of-two block is generated (preserving the net's balance) and
the first `n` points are returned. Scrambling XORs the 32-bit digits of every
coordinate with a seed-dependent random word, which keeps the balance
properties of the unscrambled net while randomizing the points.
"""
function sobol_points(dimension::Integer, n::Integer, seed::Integer)
    dimension >= 1 || throw(ArgumentError("dimension must be positive"))
    n >= 1 || throw(ArgumentError("n must be positive"))
    bits = n <= 1 ? 0 : (8 * sizeof(Int) - leading_zeros(Int(n) - 1))
    total = 1 << bits
    sequence = SobolSeq(dimension)
    points = Vector{Vector{Float64}}(undef, total)
    points[1] = zeros(dimension)
    for i in 2:total
        points[i] = next!(sequence)
    end
    shift = rand(Xoshiro(seed), UInt32, dimension)
    scale = 2.0^32
    for p in points, j in 1:dimension
        p[j] = (round(UInt32, p[j] * scale) ⊻ shift[j]) / scale
    end
    return points[1:n]
end

_point_list(x) = x === nothing ? nothing : [Vector{Float64}(p) for p in x]

function _warm_rows(warm_start, transform::BoxTransform)
    rows = if warm_start isa AbstractMatrix
        [Vector{Float64}(r) for r in eachrow(warm_start)]
    elseif warm_start isa AbstractVector{<:Real}
        [Vector{Float64}(warm_start)]
    elseif warm_start isa AbstractVector
        [Vector{Float64}(r) for r in warm_start]
    else
        throw(ArgumentError("warm_start must contain physical parameter vectors"))
    end
    all(r -> length(r) == transform.size, rows) ||
        throw(ArgumentError("warm_start must contain physical parameter vectors of length $(transform.size)"))
    return rows
end

function _choose_executor(executor, workers)
    (executor !== nothing && workers !== nothing) && throw(ArgumentError("pass either workers or executor, not both"))
    executor === nothing || (executor isa AbstractExecutor || throw(ArgumentError("executor must be an AbstractExecutor")))
    executor === nothing || return executor
    if workers !== nothing
        pids = collect(Int, workers)
        return isempty(pids) ? InlineExecutor() : DistributedExecutor(pids)
    end
    return nprocs() > 1 ? DistributedExecutor(Distributed.workers()) : InlineExecutor()
end

# Wait for in-flight jobs so their records reach the journal before we leave.
function _drain!(done::Channel, pending::Set{Int})
    while !isempty(pending)
        tag, _ = take!(done)
        delete!(pending, tag)
    end
    return nothing
end

function _screen(run_id, points, config, executor, store)
    n = length(points)
    values = Vector{Union{Nothing,Float64}}(nothing, n)
    done = Channel{Any}(Inf)
    pending = Set{Int}()
    next_index, stopped = 1, false
    try
        while !isempty(pending) || (next_index <= n && !stopped)
            while !stopped && next_index <= n && length(pending) < capacity(executor)
                i = next_index
                next_index += 1
                submit!(executor, done, i, _screen_job, run_id, i, points[i])
                push!(pending, i)
            end
            isempty(pending) && break
            tag, (flag, payload) = take!(done)
            delete!(pending, tag)
            flag === :error && throw(payload)
            values[tag] = payload
            payload === nothing && (stopped = true)
        end
    catch exc
        exc isa InterruptException || _drain!(done, pending)
        rethrow()
    end
    any(isnothing, values) && return false
    ranked = sort([i for i in 1:n if isfinite(values[i])]; by=i -> (values[i], i))
    count = config.n_local === nothing ? max(1, ceil(Int, 0.1 * config.n_samples)) : config.n_local
    put!(store, "seeds", [points[i] for i in ranked[1:min(count, end)]])
    return true
end

function _target_reached(store, config)
    config.target_value === nothing && return false
    record = best(store)
    return record !== nothing && record.value <= config.target_value
end

function _search(run_id, seeds, config, executor, store, callback, directory)
    rows = Dict(r.id => r for r in local_rows(store))
    indices = [i for i in 1:length(seeds) if !(haskey(rows, i) && rows[i].status === :done)]
    done = Channel{Any}(Inf)
    pending = Set{Int}()
    position = 1
    try
        while !isempty(pending) || position <= length(indices)
            done_results = [r.result for r in local_rows(store) if r.status === :done && r.result !== nothing]
            feasible = [r for r in done_results if r.fun !== nothing]
            incumbent = isempty(feasible) ? nothing : feasible[argmin([(r.fun, r.index) for r in feasible])]
            # Bootstrap the best Sobol seed before opening the parallel pipeline.
            cap = isempty(done_results) ? 1 : capacity(executor)
            stop = exhausted(store) || _target_reached(store, config)
            while !stop && position <= length(indices) && length(pending) < cap
                i = indices[position]
                position += 1
                if !haskey(rows, i)
                    seed = seeds[i]
                    weight = clamp((i / length(seeds))^config.mixing_power, config.mixing_min, config.mixing_max)
                    start = incumbent === nothing ? seed : (1 - weight) .* seed .+ weight .* incumbent.unit
                    create_local!(store, i, start, seed)
                end
                submit!(executor, done, i, _local_job, run_id, i)
                push!(pending, i)
            end
            isempty(pending) && break
            tag, (flag, payload) = take!(done)
            delete!(pending, tag)
            flag === :error && throw(payload)   # Unexpected model/programming errors abort visibly.
            snapshot = _result(store, directory, :running, "local search in progress")
            export_result(store, to_dict(snapshot))
            callback === nothing || callback(snapshot)
        end
    catch exc
        exc isa InterruptException || _drain!(done, pending)
        rethrow()
    end
    return nothing
end

function _result(store, directory, status::Symbol, message::AbstractString)
    record = best(store)
    rows = local_rows(store)
    return TikTakResult(
        record === nothing ? nothing : copy(record.parameters),
        record === nothing ? Inf : record.value,
        record === nothing || record.moments === nothing ? nothing : copy(record.moments),
        record === nothing || record.residuals === nothing ? nothing : copy(record.residuals),
        n_attempts(store), n_failed(store), count(r -> r.status === :done, rows),
        status, String(message), String(directory),
        [r.result for r in rows if r.result !== nothing])
end

"""
    minimize(objective, bounds; kwargs...) -> TikTakResult

Minimize a scalar objective or a [`MomentObjective`](@ref) with the TikTak
multistart algorithm (Arnoud, Guvenen & Kleineberg 2022): screen scrambled
Sobol points, retain the best seeds, and run local searches that start from a
mix of each seed and the best local optimum found so far.

`objective(θ::Vector{Float64})` returns a real number or an [`Evaluation`](@ref)
and may throw `ModelEvaluationError` for expected infeasibility. `bounds` is a
vector of `(lower, upper)` pairs; infinite bounds are allowed (see
[`BoxTransform`](@ref) for `scale`, `location`, and `tail`).

Keyword arguments:
- `config::TikTakConfig`: algorithm settings.
- `run_dir`: directory for the journal, result snapshot and lock. A temporary
  directory is used when omitted. With an explicit `run_dir`, a nonempty
  versioned `problem_id` is required; it must change whenever the model code,
  targets, solver accuracy, input data, or simulation draws change.
- `resume=true`: continue a run in `run_dir` after all previous workers have
  stopped. The specification must match except for `max_evals` and `max_seconds`.
- `warm_start`: physical parameter vectors (or a matrix of rows) that join the
  screening pool of a NEW run and are reevaluated; see [`load_estimates`](@ref).
- `workers`: worker process ids to use. By default all `Distributed.workers()`
  are used, or the coordinator alone when there are none. Pass `Int[]` to force
  serial execution.
- `executor`: an [`AbstractExecutor`](@ref) instead of `workers`.
- `callback(result)`: called on the coordinator after each completed local search.

`max_evals` is a hard cumulative cap on model-call attempts (including failed
and interrupted ones, across resumes; cached evaluations are free).
`max_seconds` is a soft deadline: running model calls finish and are recorded.
"""
function minimize(objective, bounds; config::TikTakConfig=TikTakConfig(), scale=nothing, location=nothing,
                  tail=1e-6, run_dir=nothing, problem_id=nothing, resume::Bool=false, warm_start=nothing,
                  workers=nothing, executor=nothing, callback=nothing)
    transform = BoxTransform(bounds; scale=scale, location=location, tail=tail)
    if run_dir === nothing
        resume && throw(ArgumentError("resume requires run_dir"))
        directory = mktempdir(; prefix="tiktak-", cleanup=false)
        problem_id = something(problem_id, "temporary-run")
    else
        directory = abspath(String(run_dir))
        (problem_id isa AbstractString && !isempty(strip(problem_id))) ||
            throw(ArgumentError("provide a nonempty versioned problem_id with run_dir"))
    end
    (resume && warm_start !== nothing) &&
        throw(ArgumentError("warm_start is for new runs; resume reuses saved screening points"))
    spec = Dict{String,Any}(
        "schema" => 1, "problem_id" => String(problem_id), "transform" => specification(transform),
        "algorithm" => specification(config),
        "moments" => objective isa MomentObjective ? specification(objective) : nothing)
    # Validate serializability before starting expensive work.
    spec = JSON.parse(JSON.json(spec); dicttype=Dict{String,Any})
    deadline = config.max_seconds === nothing ? nothing : time() + config.max_seconds
    executor = _choose_executor(executor, workers)
    run_id = string(getpid(), "-", time_ns())

    handle = coordinator_lock(directory)
    store = Store(directory)
    registered = installed = false
    try
        initialize!(store, spec; resume=resume, max_evals=config.max_evals, deadline=deadline)
        points = _point_list(get(store, "points"))
        if points === nothing
            points = transform.dimension > 0 ? sobol_points(transform.dimension, config.n_samples, config.seed) :
                     [Float64[]]
            if warm_start !== nothing
                append!(points, to_unit(transform, x) for x in _warm_rows(warm_start, transform))
            end
            # Identical fixed/warm points need only one screening task.
            points = unique(_canonical.(points))
            put!(store, "points", points)
        end
        _register_store!(run_id, store)
        registered = true
        setup!(executor, run_id, objective, transform, config, store)
        installed = true
        seeds = _point_list(get(store, "seeds"))
        screened = seeds !== nothing || _screen(run_id, points, config, executor, store)
        seeds = _point_list(get(store, "seeds"))
        if screened && seeds !== nothing && !isempty(seeds) && transform.dimension > 0
            _search(run_id, seeds, config, executor, store, callback, directory)
        end
        n_done = count(r -> r.status === :done, local_rows(store))
        status, message = :completed, "all retained seeds processed; global optimality is not certified"
        if best(store) === nothing
            status, message = :no_feasible_point, "no finite model evaluation found within the available budget"
        elseif _target_reached(store, config)
            status, message = :target_reached, "requested objective target attained"
        elseif !screened || (transform.dimension > 0 && n_done < length(seeds))
            status, message = :budget_exhausted, "evaluation or wall-clock budget exhausted; increase budget and resume"
        end
        result = _result(store, directory, status, message)
        export_result(store, to_dict(result))
        return result
    finally
        installed && teardown!(executor, run_id)
        registered && _unregister_store!(run_id)
        close(store)
        close(handle)
    end
end
