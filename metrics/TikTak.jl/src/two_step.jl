# Two-step minimum-distance estimation: a first search under the objective's own
# weights, then a separately journaled search under inverse-covariance weights.

"""
    TwoStepResult

Outcome of [`two_step`](@ref). `x` and `fun` are the second-step estimate and loss
(`nothing` and `Inf` until the second step runs); `first` and `second` are the
[`TikTakResult`](@ref)s of the two searches. `objective` is the second-step
[`MomentObjective`](@ref), `weights` its [`InverseCovarianceWeights`](@ref), and
`covariance`, `nobs`, `simulation_covariance`, and `simulation_nobs` the moment covariance
they were built from (the simulation fields are `nothing` without a separate simulation
contribution; `n_simulations = R` is stored as `Σₛ = Σ`, `nₛ = R * nobs`, and in
`n_simulations`). Pass `result` itself to [`parameter_covariance`](@ref) to use them. `j_statistic` is the minimized second-step loss when the
weights are unregularized and the second search finished, otherwise `NaN`;
`degrees_of_freedom` is the number of moments minus free parameters. `status` and
`message` come from the step that ran last.
"""
struct TwoStepResult
    x::Union{Nothing,Vector{Float64}}
    fun::Float64
    first::TikTakResult
    second::Union{Nothing,TikTakResult}
    objective::Union{Nothing,MomentObjective}
    weights::Union{Nothing,InverseCovarianceWeights}
    covariance::Union{Nothing,Matrix{Float64}}
    nobs::Float64
    simulation_covariance::Union{Nothing,Matrix{Float64}}
    simulation_nobs::Union{Nothing,Float64}
    n_simulations::Union{Nothing,Float64}
    j_statistic::Float64
    degrees_of_freedom::Int
    status::Symbol
    message::String
    run_dir::String
end

has_solution(r::TwoStepResult) = r.x !== nothing && isfinite(r.fun)

# Version tag of a second-step run: the criterion changes exactly when its weights do.
function _weights_tag(objectives...)
    values = reduce(vcat, [vec(o.weights) for o in objectives if o !== nothing])
    return bytes2hex(sha256(reinterpret(UInt8, _canonical(values))))[1:12]
end

_initialized(directory) = isfile(joinpath(directory, JOURNAL_NAME)) &&
    get(Store(directory; readonly=true), "specification") !== nothing

"""
    two_step(objective::MomentObjective, bounds; covariance, nobs, kwargs...) -> TwoStepResult

Two-step minimum-distance estimation. The first step minimizes `objective` with its own
weights. The second step minimizes the same moments with
[`inverse_covariance_weights`](@ref) of `covariance`, in a new run whose screening pool
includes the first-step estimate and other distinct first-step minima. Only parameter
vectors pass between the steps; every second-step loss is computed under the
second-step criterion.

- `covariance`: `Σ` with the conventions of [`parameter_covariance`](@ref), either a
  matrix or a function `θ -> Σ` evaluated once at the first-step estimate, for example
  from moment contributions simulated there with fixed draws. A matrix is validated
  before the first step. With a fixed matrix the second-step criterion does not depend
  on the first step, which then only supplies starting points.
- `nobs`: the sample size of `Σ`. Second-step losses are on the scale of the J statistic.
- `simulation_covariance` with `simulation_nobs`, or `n_simulations`: the simulation noise
  in the model moments, with the conventions of [`parameter_covariance`](@ref). Like
  `covariance`, `simulation_covariance` may be a function `θ -> Σₛ` evaluated once at the
  first-step estimate. Omit them when `Σ` already includes simulation noise or the model
  moments are computed without simulation.
- `weight_options=(;)`: keywords for [`inverse_covariance_weights`](@ref), for example
  `(; max_condition=1e4)`, `(; diagonal=true)`, or `moment_names`.
- `config`: first-step [`TikTakConfig`](@ref). `second_config` defaults to `config`
  without `target_value`, since the losses differ in scale; adjust `f_tol` for the same
  reason, or lower `n_samples` and `n_local` for a cheaper refinement.
- `n_warm=5`: number of second-step starting points taken from the first step: its
  estimate, then other minima of [`search_summary`](@ref) in order of first-step loss.
- `run_dir`, `problem_id`, `resume`: as in [`minimize`](@ref). The first step is saved in
  `run_dir/step1` and the second in `run_dir/step2-<tag>`, where the tag hashes the
  second-step weights. If the weights change on a resume, for example because `covariance`
  depends on draws that are not fixed, a new second-step run starts instead of reusing
  losses from another criterion.
- `screen_objective`: a cheaper [`MomentObjective`](@ref) for screening (see
  [`minimize`](@ref)), reweighted by the same inverse covariance in the second step.
  Scalar screening objectives are rejected because they cannot be reweighted.
- `warm_start`: starting points of the first step, which must be a new run.
- Other keywords, such as `scale`, `location`, `tail`, `workers`, `executor`, `callback`,
  and `store_evaluations`, are passed to both calls of [`minimize`](@ref).

The second step starts only when the first ends with status `:completed` or
`:target_reached`. Otherwise `second === nothing` and the first step's status is
returned; increase its budget and resume.

Under correct specification, with consistent covariances (including simulation noise) and
unregularized weights, `j_statistic` is asymptotically χ² with `degrees_of_freedom`
degrees of freedom; the search does not certify that it is at the global minimum. With a
`covariance` evaluated at the first-step estimate, `Σ` can be reevaluated at the
second-step estimate for standard errors.
"""
function two_step(objective::MomentObjective, bounds; covariance, nobs::Real, simulation_covariance=nothing,
                  simulation_nobs=nothing, n_simulations=nothing, weight_options::NamedTuple=(;),
                  config::TikTakConfig=TikTakConfig(),
                  second_config::TikTakConfig=TikTakConfig(config; target_value=nothing),
                  n_warm::Integer=5, run_dir=nothing, problem_id=nothing, resume::Bool=false,
                  screen_objective=nothing, warm_start=nothing, kwargs...)
    (screen_objective === nothing || screen_objective isa MomentObjective) || throw(ArgumentError(
        "two_step requires screen_objective to be a MomentObjective so that it can be reweighted"))
    _positive_int(n_warm) || throw(ArgumentError("n_warm must be a positive integer"))
    weigh(sigma, simulation) = inverse_covariance_weights(sigma; nobs=nobs, simulation_covariance=simulation,
                                                          simulation_nobs=simulation_nobs,
                                                          n_simulations=n_simulations, weight_options...)
    _fixed(c) = c === nothing || c isa AbstractMatrix
    if covariance isa AbstractMatrix && _fixed(simulation_covariance)
        # Fail before the first search if a fixed covariance cannot be inverted or does not fit.
        MomentObjective(objective; weights=weigh(covariance, simulation_covariance))
    end
    if run_dir === nothing
        resume && throw(ArgumentError("resume requires run_dir"))
        directory = mktempdir(; prefix="tiktak-two-step-", cleanup=false)
        problem_id = something(problem_id, "temporary-run")
    else
        directory = abspath(String(run_dir))
        (problem_id isa AbstractString && !isempty(strip(problem_id))) ||
            throw(ArgumentError("provide a nonempty versioned problem_id with run_dir"))
    end
    lower, upper = _parse_bounds(bounds)
    degrees_of_freedom = length(objective.target) - count(lower .!= upper)

    first_dir = joinpath(directory, "step1")
    first = minimize(objective, bounds; config=config, run_dir=first_dir, problem_id="$problem_id/step1",
                     resume=resume, screen_objective=screen_objective, warm_start=warm_start, kwargs...)
    if !(first.status in (:completed, :target_reached))
        return TwoStepResult(nothing, Inf, first, nothing, nothing, nothing, nothing, Float64(nobs), nothing,
                             nothing, nothing, NaN,
                             degrees_of_freedom, first.status, "first step: " * first.message, directory)
    end

    sigma = covariance isa AbstractMatrix ? covariance : covariance(copy(first.x))
    sigma isa AbstractMatrix{<:Real} || throw(ArgumentError("covariance(θ) must return a real matrix"))
    simulation = _fixed(simulation_covariance) ? simulation_covariance : simulation_covariance(copy(first.x))
    (simulation === nothing || simulation isa AbstractMatrix{<:Real}) ||
        throw(ArgumentError("simulation_covariance(θ) must return a real matrix"))
    weights = weigh(sigma, simulation)
    # Stored explicitly, so that `n_simulations = R` reads Σₛ = Σ with nₛ = R * nobs.
    stored_simulation, stored_nobs = n_simulations !== nothing ? (sigma, n_simulations * nobs) :
        simulation === nothing ? (nothing, nothing) : (simulation, simulation_nobs)
    second_objective = MomentObjective(objective; weights=weights)
    second_screen = screen_objective === nothing ? nothing : MomentObjective(screen_objective; weights=weights)
    second_dir = joinpath(directory, "step2-" * _weights_tag(second_objective, second_screen))
    second_resume = resume && _initialized(second_dir)
    starts = nothing
    if !second_resume
        minima = search_summary(first_dir).minima
        starts = unique([[first.x]; [m.x for m in minima]])[1:min(n_warm, end)]
    end
    second = minimize(second_objective, bounds; config=second_config, run_dir=second_dir,
                      problem_id="$problem_id/step2", resume=second_resume, screen_objective=second_screen,
                      warm_start=starts, kwargs...)

    finished = has_solution(second) && second.status in (:completed, :target_reached)
    j_statistic = finished && weights.shrinkage == 0 ? second.fun : NaN
    return TwoStepResult(second.x, second.fun, first, second, second_objective, weights, Matrix{Float64}(sigma),
                         Float64(nobs), stored_simulation === nothing ? nothing : Matrix{Float64}(stored_simulation),
                         stored_nobs === nothing ? nothing : Float64(stored_nobs),
                         n_simulations === nothing ? nothing : Float64(n_simulations), j_statistic, degrees_of_freedom,
                         second.status,
                         "second step: " * second.message, directory)
end

"""
    parameter_covariance(jac, result::TwoStepResult; kwargs...) -> ParameterCovariance

Sandwich covariance under the second-step objective and the moment covariance of
`result`, including its simulation contribution. `jac` must be evaluated at `result.x`.
Keywords are passed to [`parameter_covariance`](@ref) and override the stored values, for
example `covariance` reevaluated at the second-step estimate. With `n_simulations` stored,
the simulation term follows an overriding `covariance`; passing any of
`simulation_covariance`, `simulation_nobs`, or `n_simulations` replaces the stored
simulation term entirely.
"""
function parameter_covariance(jac::MomentJacobian, result::TwoStepResult; kwargs...)
    result.objective === nothing && throw(ArgumentError("the second step of this two-step estimate has not run"))
    isapprox(jac.parameters, result.x; rtol=1e-12, atol=1e-14) ||
        throw(ArgumentError("jac must be evaluated at the second-step estimate result.x"))
    stored = result.n_simulations !== nothing ? (; n_simulations=result.n_simulations) :
             result.simulation_covariance === nothing ? (;) :
             (; simulation_covariance=result.simulation_covariance, simulation_nobs=result.simulation_nobs)
    any(in(keys(kwargs)), (:simulation_covariance, :simulation_nobs, :n_simulations)) && (stored = (;))
    return parameter_covariance(jac, result.objective; covariance=result.covariance, nobs=result.nobs, stored...,
                                kwargs...)
end

_point_string(x) = x === nothing ? "none" : "[" * join(_short.(x), ", ") * "]"

function Base.show(io::IO, ::MIME"text/plain", r::TwoStepResult)
    print(io, "TwoStepResult: ", r.status, "; ", r.message,
          "\n  first step:  loss ", _short(r.first.fun), " at ", _point_string(r.first.x))
    if r.second !== nothing
        print(io, "\n  second step: loss ", _short(r.second.fun), " at ", _point_string(r.second.x),
              "\n  weights: ", r.weights)
        isnan(r.j_statistic) ||
            print(io, "\n  J statistic ", _short(r.j_statistic), " with ", r.degrees_of_freedom, " degrees of freedom")
    end
    print(io, "\n  run directory: ", r.run_dir)
end

Base.show(io::IO, r::TwoStepResult) =
    print(io, "TwoStepResult(status=:", r.status, ", fun=", r.fun, ", x=", r.x, ")")
