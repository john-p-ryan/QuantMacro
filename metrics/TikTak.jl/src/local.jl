# Failure-aware evaluation and local search workers; no model-specific logic lives here.

# --- Local search methods ---------------------------------------------------

"""
    LocalMethod

Abstract supertype of local search methods. A method `m` is used through
`run_local_search(m, f, start, config) -> (converged::Bool, message::String)`,
where `f(u::Vector{Float64}) -> Float64` evaluates the objective at a point of
the unit box (returning `Inf` for failed evaluations) and `start` lies in the
box. Implementations must respect the box `[0, 1]^d`, must let
`BudgetExhausted`/`LocalBudgetExhausted` exceptions propagate, and need not
return the best point: it is recovered from the evaluation history.
"""
abstract type LocalMethod end

"""
    NelderMeadLocal()

Bounded adaptive Nelder–Mead simplex (Gao & Han 2012 coefficients) with an
explicit full-rank inward initial simplex of edge `initial_step`, trial points
clipped to the unit box, and joint `x_tol`/`f_tol` termination. Handles failed
(infinite) evaluations gracefully; simplex collapse at a bound and poor scaling
in high dimensions remain possible. Comparable to the paper's TikTak-nm.
"""
struct NelderMeadLocal <: LocalMethod end

"""
    PatternSearchLocal()

Opportunistic coordinate polling with mesh contraction from `initial_step` to
`x_tol`; `f_tol` is the absolute improvement threshold. No derivatives or fitted
surrogate, so it copes with jumps and invalid regions, but it can stop on
plateaus or miss diagonal descent directions. This is not MADS.
"""
struct PatternSearchLocal <: LocalMethod end

"""
    NLoptLocal(algorithm=:LN_BOBYQA; options...)

Derivative-free NLopt algorithm with bound constraints (for example
`:LN_BOBYQA`, `:LN_COBYLA`, `:LN_SBPLX`, `:LN_NELDERMEAD`). Requires
`using NLopt` on every process. `initial_step`, `x_tol` (absolute), `f_tol`
(absolute) and `local_max_evals` are taken from the configuration; `options`
are set on the `NLopt.Opt` object afterwards (for example `xtol_rel=1e-8`).
BOBYQA's quadratic models can be undermined by infinite (failed) evaluations
and by simulation noise; compare with `NelderMeadLocal` on a pilot budget.
"""
struct NLoptLocal <: LocalMethod
    algorithm::Symbol
    options::Dict{Symbol,Any}
end
NLoptLocal(algorithm::Symbol=:LN_BOBYQA; options...) = NLoptLocal(algorithm, Dict{Symbol,Any}(options))

"""
    CustomLocal(solve; name="custom")

Plug in any local solver: `solve(f, start, config) -> (converged, message)`
with the contract described under [`LocalMethod`](@ref). `name` is recorded in
the run specification and must change when the solver changes.
"""
struct CustomLocal{F} <: LocalMethod
    solve::F
    name::String
end
CustomLocal(solve; name="custom") = CustomLocal(solve, String(name))

specification(::NelderMeadLocal) = Dict{String,Any}("method" => "NelderMeadLocal")
specification(::PatternSearchLocal) = Dict{String,Any}("method" => "PatternSearchLocal")
specification(m::CustomLocal) = Dict{String,Any}("method" => "CustomLocal", "name" => m.name)
function specification(m::NLoptLocal)
    options = Dict{String,Any}(string(k) => repr(v) for (k, v) in m.options)
    return Dict{String,Any}("method" => "NLoptLocal", "algorithm" => string(m.algorithm), "options" => options)
end

# Filled in by the NLopt package extension when NLopt is loaded on this process.
const NLOPT_BACKEND = Ref{Any}(nothing)

function run_local_search(method::NLoptLocal, f, start, config)
    backend = NLOPT_BACKEND[]
    backend === nothing && throw(ErrorException(
        "NLoptLocal requires `using NLopt` on every process (it activates the TikTak NLopt extension)"))
    return backend(method, f, start, config)
end
run_local_search(m::CustomLocal, f, start, config) = m.solve(f, start, config)

function run_local_search(::NelderMeadLocal, f, start, config)
    return nelder_mead(f, start; initial_step=config.initial_step, x_tol=config.x_tol,
                       f_tol=config.f_tol, max_calls=config.local_max_evals)
end

function run_local_search(::PatternSearchLocal, f, start, config)
    return pattern_search(f, start; step=config.initial_step, tolerance=config.x_tol,
                          improvement_tol=config.f_tol, max_calls=config.local_max_evals)
end

"""
    pattern_search(f, start; step, tolerance, improvement_tol, max_calls)

Opportunistic coordinate polling with mesh contraction on the unit box.
"""
function pattern_search(f, start::AbstractVector; step, tolerance, improvement_tol, max_calls)
    x = Vector{Float64}(start)
    best = f(x)
    calls = 1
    while step > tolerance && calls < max_calls
        improved = false
        for j in eachindex(x)
            for direction in (1, -1)
                trial = copy(x)
                trial[j] = clamp(x[j] + direction * step, 0.0, 1.0)
                trial == x && continue
                value = f(trial)
                calls += 1
                if value < best - improvement_tol
                    x, best, improved = trial, value, true
                    break
                end
                calls >= max_calls && break
            end
            (improved || calls >= max_calls) && break
        end
        improved || (step *= 0.5)
    end
    converged = step <= tolerance
    return converged, converged ? "mesh tolerance reached" : "local call limit reached"
end

"""
    nelder_mead(f, start; initial_step, x_tol, f_tol, max_calls)

Bounded adaptive Nelder–Mead on the unit box. Stops when every vertex is within
`x_tol` of the best vertex and every value within `f_tol` of the best value, or
after about `max_calls` objective calls.
"""
function nelder_mead(f, start::AbstractVector; initial_step, x_tol, f_tol, max_calls)
    n = length(start)
    n >= 1 || throw(ArgumentError("nelder_mead requires at least one free parameter"))
    x0 = Vector{Float64}(start)
    edge = min(initial_step, 0.5)
    # Build a full-rank inward simplex; a relative simplex is tiny near zero and
    # can collapse at a bound.
    simplex = [copy(x0)]
    for j in 1:n
        vertex = copy(x0)
        vertex[j] = clamp(vertex[j] + (vertex[j] <= 0.5 ? edge : -edge), 0.0, 1.0)
        push!(simplex, vertex)
    end
    calls = 0
    evaluate = x -> (calls += 1; f(x))
    values = [evaluate(v) for v in simplex]
    if n >= 2
        α, β, γ, δ = 1.0, 1 + 2 / n, 0.75 - 1 / (2n), 1 - 1 / n
    else
        α, β, γ, δ = 1.0, 2.0, 0.5, 0.5
    end
    while true
        order = sortperm(values)
        simplex, values = simplex[order], values[order]
        xdiff = maximum(maximum(abs, simplex[i] .- simplex[1]) for i in 2:n+1)
        fdiff = maximum(abs(values[i] - values[1]) for i in 2:n+1)
        (xdiff <= x_tol && fdiff <= f_tol) && return true, "simplex tolerance reached"
        calls >= max_calls && return false, "local call limit reached"
        centroid = sum(simplex[1:n]) ./ n
        worst = simplex[end]
        xr = clamp.(centroid .+ α .* (centroid .- worst), 0.0, 1.0)
        fr = evaluate(xr)
        if fr < values[1]
            xe = clamp.(centroid .+ β .* (xr .- centroid), 0.0, 1.0)
            fe = evaluate(xe)
            simplex[end], values[end] = fe < fr ? (xe, fe) : (xr, fr)
        elseif fr < values[n]
            simplex[end], values[end] = xr, fr
        else
            shrink = true
            if fr < values[end]
                xc = clamp.(centroid .+ γ .* (xr .- centroid), 0.0, 1.0)
                fc = evaluate(xc)
                if fc <= fr
                    simplex[end], values[end], shrink = xc, fc, false
                end
            else
                xcc = clamp.(centroid .- γ .* (centroid .- worst), 0.0, 1.0)
                fcc = evaluate(xcc)
                if fcc < values[end]
                    simplex[end], values[end], shrink = xcc, fcc, false
                end
            end
            if shrink
                for i in 2:n+1
                    simplex[i] = clamp.(simplex[1] .+ δ .* (simplex[i] .- simplex[1]), 0.0, 1.0)
                    values[i] = evaluate(simplex[i])
                end
            end
        end
    end
end

# --- Store access from workers ------------------------------------------------

"""
    RemoteStore(pid, run_id)

Proxy for the coordinator's `Store`: every operation becomes a
`remotecall_fetch` to process `pid`, which looks up the active run `run_id`.
"""
struct RemoteStore
    pid::Int
    run_id::String
end

const ACTIVE_STORES = Dict{String,Store}()
const ACTIVE_STORES_LOCK = ReentrantLock()

function _register_store!(run_id::AbstractString, store::Store)
    lock(ACTIVE_STORES_LOCK) do
        ACTIVE_STORES[String(run_id)] = store
    end
    return nothing
end

function _unregister_store!(run_id::AbstractString)
    lock(ACTIVE_STORES_LOCK) do
        delete!(ACTIVE_STORES, String(run_id))
    end
    return nothing
end

function _active_store(run_id::AbstractString)
    lock(ACTIVE_STORES_LOCK) do
        store = get(ACTIVE_STORES, String(run_id), nothing)
        store === nothing && throw(ErrorException("no active TikTak run $run_id on process $(myid())"))
        return store
    end
end

_remote_op(run_id, op, args, kwargs) = op(_active_store(run_id), args...; kwargs...)

for op in (:claim!, :finish!, :abandon!, :best, :local_row, :finish_local!)
    @eval function $op(h::RemoteStore, args...; kwargs...)
        return remotecall_fetch(_remote_op, h.pid, h.run_id, $op, args, (; kwargs...))
    end
end

# --- Evaluator --------------------------------------------------------------

"""
    Evaluator(objective, transform, store, task, config; islocal=false)

Callable mapping a unit-box point to an objective value while recording every
model call in the store: it claims a budget slot (or returns a cached outcome),
runs the model, classifies failures, and stores moments, residuals, errors and
runtime. Failed evaluations return `Inf`.
"""
struct Evaluator{O,S}
    objective::O
    transform::BoxTransform
    store::S
    task::String
    limit::Union{Nothing,Int}
    failure_exceptions::Tuple
end

function Evaluator(objective, transform::BoxTransform, store, task::AbstractString, config; islocal::Bool=false)
    return Evaluator(objective, transform, store, String(task), islocal ? config.local_max_evals : nothing,
                     config.failure_exceptions)
end

function _claim_point(store, unit, parameters, task, limit)
    while true
        status, key, payload = claim!(store, unit, parameters, task, limit)
        status === :claimed && return key, nothing
        status === :ok && return key, Float64(payload)
        status === :failed && return key, Inf
        status === :error && throw(ErrorException("cached unexpected model error: " * String(payload)))
        status === :budget && throw(BudgetExhausted())
        status === :local_budget && throw(LocalBudgetExhausted())
        # Another worker is evaluating this exact point: no second model call and
        # no second budget charge. Budgets are rechecked while waiting.
        sleep(0.05)
    end
end

_describe(exc) = string(nameof(typeof(exc)), ": ", sprint(showerror, exc))
_is_failure(exc, types) = any(T -> exc isa T, types)

function (ev::Evaluator)(unit_in::AbstractVector)
    # Supported optimizers respect bounds; tolerate only floating-point drift.
    unit = Vector{Float64}(unit_in)
    (any(<(-1e-12), unit) || any(>(1 + 1e-12), unit)) &&
        throw(ArgumentError("local optimizer proposed a point outside the unit box"))
    unit = _canonical(clamp.(unit, 0.0, 1.0))
    parameters = to_parameters(ev.transform, unit)
    key, cached = _claim_point(ev.store, unit, parameters, ev.task, ev.limit)
    cached === nothing || return cached
    started = time_ns()
    local evaluation
    try
        output = ev.objective(copy(parameters))
        if output isa Evaluation
            evaluation = output
        elseif output isa Real
            evaluation = Evaluation(Float64(output))
        else
            throw(ArgumentError("objective must return a real number or an Evaluation"))
        end
        isfinite(evaluation.value) || throw(ModelEvaluationError("objective returned a nonfinite value"))
        evaluation.moments === nothing || all(isfinite, evaluation.moments) ||
            throw(ModelEvaluationError("nonfinite moments"))
        evaluation.residuals === nothing || all(isfinite, evaluation.residuals) ||
            throw(ModelEvaluationError("nonfinite residuals"))
    catch exc
        seconds = (time_ns() - started) / 1e9
        if exc isa InterruptException
            # Interruptions are retryable on resume, but retain the budget charge.
            abandon!(ev.store, key)
            rethrow()
        elseif _is_failure(exc, ev.failure_exceptions)
            finish!(ev.store, key; error=_describe(exc), seconds=seconds)
            return Inf
        else
            finish!(ev.store, key; error=_describe(exc), seconds=seconds, unexpected=true)
            rethrow()
        end
    end
    finish!(ev.store, key; value=evaluation.value, moments=evaluation.moments,
            residuals=evaluation.residuals, seconds=(time_ns() - started) / 1e9)
    return evaluation.value
end

"""
    evaluate_point(objective, transform, store, unit, task, config)

Screen one point. Returns its value (`Inf` when failed) or `nothing` when the
global budget is exhausted.
"""
function evaluate_point(objective, transform, store, unit, task, config)
    evaluator = Evaluator(objective, transform, store, task, config)
    try
        return evaluator(unit)
    catch exc
        exc isa BudgetExhausted && return nothing
        rethrow()
    end
end

"""
    run_local(objective, transform, store, index, config) -> LocalResult

Run the local search recorded under `index`: evaluate its (mixed) start, fall
back to the unmixed Sobol seed if the start is infeasible, run the configured
local method, and record the best evaluated point of the task.
"""
function run_local(objective, transform, store, index::Integer, config)
    task = "local:$index"
    evaluator = Evaluator(objective, transform, store, task, config; islocal=true)
    row = local_row(store, index)
    start, seed = copy(row.start), copy(row.seed)
    converged, complete, message = false, true, ""
    try
        # The mixing segment can pass through an economic infeasibility hole.
        # The unblended Sobol seed is known feasible and is a safe fallback.
        isfinite(evaluator(start)) || (start = seed)
        converged, message = run_local_search(config.local_method, evaluator, start, config)
    catch exc
        if exc isa LocalBudgetExhausted
            message = "local evaluation budget exhausted"
        elseif exc isa BudgetExhausted
            complete, message = false, "global evaluation or wall-clock budget exhausted"
        else
            rethrow()
        end
    end
    record = best(store, task)
    result = LocalResult(Int(index), record === nothing ? nothing : copy(record.unit),
                         record === nothing ? nothing : copy(record.parameters),
                         record === nothing ? nothing : record.value,
                         converged && record !== nothing, String(message))
    finish_local!(store, index, result, complete)
    return result
end
