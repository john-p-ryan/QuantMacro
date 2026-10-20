# Local identification diagnostics and sandwich inference for minimum-distance estimates.
#
# Everything here linearizes the moment function at a given estimate. The results
# are local: they assume fixed simulation draws, a correctly specified covariance
# of moment errors, and an interior optimum unless flagged otherwise.

"""
    MomentJacobian

Derivatives `jacobian[i, k] = ∂mᵢ/∂θⱼ` of the raw model moments at `parameters`,
in original parameter units, for the free parameters `j = free[k]`. `steps` (the
largest offset actually used), `schemes` (`:central`, `:forward`, `:backward`,
`:failed`, or `:supplied`) and `sensitivity` describe each column; `failures`
lists failed perturbations.
"""
struct MomentJacobian
    jacobian::Matrix{Float64}
    parameters::Vector{Float64}
    free::Vector{Int}
    names::Vector{String}
    lower::Vector{Float64}
    upper::Vector{Float64}
    steps::Vector{Float64}
    schemes::Vector{Symbol}
    sensitivity::Vector{Float64}
    failures::Vector{String}
end

_moment_function(objective::MomentObjective) = objective.model
_moment_function(model) = model
_moment_scales(objective::MomentObjective, n_moments) = objective.scales
_moment_scales(_, n_moments) = ones(n_moments)

# A failed model call returns its description instead of moments.
function _try_moments(f, parameters, failure_exceptions)
    try
        raw = f(copy(parameters))
        raw isa AbstractVector{<:Real} || throw(ArgumentError("model must return a real vector of moments"))
        all(isfinite, raw) || return "nonfinite moments"
        return Vector{Float64}(raw)
    catch exc
        _is_failure(exc, failure_exceptions) || rethrow()
        return _describe(exc)
    end
end

# Perturbed values of `x` for a difference of size `h` that stays within [lower, upper].
function _perturbations(x, lower, upper, h)
    up, down = x + h <= upper, x - h >= lower
    up && down && return [x + h, x - h]
    up && return [x + h]
    down && return [x - h]
    # The interval is narrower than the step: difference across the wider side.
    return [upper - x >= x - lower ? upper : lower]
end

function _difference(base, plus, minus)
    plus_ok, minus_ok = plus !== nothing && plus[1] isa Vector, minus !== nothing && minus[1] isa Vector
    plus_ok && minus_ok && return (plus[1] .- minus[1]) ./ (plus[2] - minus[2]), :central
    plus_ok && return (plus[1] .- base) ./ plus[2], :forward
    minus_ok && return (minus[1] .- base) ./ minus[2], :backward
    return fill(NaN, length(base)), :failed
end

"""
    moment_jacobian(model, parameters, bounds; kwargs...) -> MomentJacobian

Jacobian of the raw model moments with respect to the free parameters, in
original parameter units. `model` is a [`MomentObjective`](@ref) or a function
`θ -> moments`; `bounds` are those passed to [`minimize`](@ref), and parameters
with equal bounds are omitted.

Supply derivatives with `jacobian`, a moments × parameters matrix or a function
of `θ` returning one (columns of fixed parameters are dropped). Otherwise finite
differences are used: central where `θⱼ ± h` lies within the bounds and
first-order one-sided otherwise. A failed perturbation (an exception in `failure_exceptions` or
nonfinite moments) falls back to the one-sided difference on the other side and
is listed in `failures`; if both sides fail the column is `NaN`.

- `step`: absolute step in parameter units (scalar, or a vector over all
  parameters). The default `cbrt(eps()) * max(|θⱼ|, 1)` suits smooth models;
  simulated moments with kinks or discrete choices need steps that are large
  relative to that granularity.
- `check_factors=(0.5, 2.0)`: the Jacobian is recomputed at these multiples of
  `step`. `sensitivity[k]` is the largest change of column `k` relative to the
  larger column norm (0 when identical, at most 2), with moments divided by the
  objective's `scales`; `NaN` when not checked. Pass `()` to skip.
- `mapper=map`: evaluates the perturbed points. Pass `pmap` to use workers, or
  `(f, xs) -> pmap(f, CachingPool(workers()), xs)` when the model carries
  large data.
- `names`: parameter names (over all parameters) used by the reports.

The model must use the same simulation draws on every call. The moments at
`parameters` are evaluated again first and last in the batch, and an error is
thrown if they differ.
"""
function moment_jacobian(model, parameters::AbstractVector{<:Real}, bounds;
                         jacobian=nothing, step=nothing, check_factors=(0.5, 2.0), mapper=map,
                         names=nothing, failure_exceptions=(ModelEvaluationError,))
    lower, upper = _parse_bounds(bounds)
    θ = Vector{Float64}(parameters)
    length(θ) == length(lower) || throw(ArgumentError("parameters must have one entry per bound"))
    (all(isfinite, θ) && all(lower .<= θ .<= upper)) ||
        throw(ArgumentError("parameters must be finite and within bounds"))
    free = findall(lower .!= upper)
    isempty(free) && throw(ArgumentError("there are no free parameters to differentiate"))
    n_free = length(free)
    labels = _labels(names, length(θ), "θ", "names")[free]
    lower, upper = lower[free], upper[free]

    if jacobian !== nothing
        supplied = Matrix{Float64}(jacobian isa AbstractMatrix ? jacobian : jacobian(copy(θ)))
        (size(supplied, 2) == length(θ) && all(isfinite, supplied)) ||
            throw(ArgumentError("supplied Jacobian must be a finite moments × parameters matrix"))
        return MomentJacobian(supplied[:, free], θ, free, labels, lower, upper, fill(NaN, n_free),
                              fill(:supplied, n_free), fill(NaN, n_free), String[])
    end

    nominal = step === nothing ? cbrt(eps()) .* max.(abs.(θ[free]), 1.0) :
        _broadcast_vector(step, NaN, length(θ), "step")[free]
    (all(isfinite, nominal) && all(>(0), nominal)) || throw(ArgumentError("steps must be positive and finite"))
    factors = (1.0, Float64.(check_factors)...)
    (all(isfinite, factors) && all(>(0), factors)) || throw(ArgumentError("check_factors must be positive"))

    f = _moment_function(model)
    base = _try_moments(f, θ, failure_exceptions)
    base isa Vector || throw(ArgumentError("model evaluation failed at the parameters: $base"))
    n_moments = length(base)
    scales = _moment_scales(model, n_moments)
    length(scales) == n_moments || throw(ArgumentError("model moments have a different length than target"))

    # Tag (factor, column, side) → perturbed point. Repeating the base point first
    # and last checks that the model is deterministic, also on workers under `pmap`.
    perturbed = [_perturbations(θ[free[k]], lower[k], upper[k], factor * nominal[k])
                 for factor in factors, k in 1:n_free]
    tags, points = Tuple{Int,Int,Int}[], [θ]
    for a in eachindex(factors), k in 1:n_free, value in perturbed[a, k]
        point = copy(θ)
        point[free[k]] = value
        push!(tags, (a, k, value > θ[free[k]] ? 1 : -1))
        push!(points, point)
    end
    push!(points, θ)
    outcomes = collect(mapper(p -> _try_moments(f, p, failure_exceptions), points))
    for repeated in (outcomes[1], outcomes[end])
        (repeated isa Vector && isapprox(repeated ./ scales, base ./ scales; rtol=1e-10)) || throw(ArgumentError(
            "model moments are not reproducible at the parameters; fix simulation draws and inner tolerances"))
    end

    # Actual offsets (θⱼ + h) - θⱼ avoid representation error in the step.
    results = Dict(tag => (outcomes[i+1], points[i+1][free[tag[2]]] - θ[free[tag[2]]])
                   for (i, tag) in enumerate(tags))
    failures = [string(labels[k], " at offset ", _short(offset), ": ", outcome)
                for ((_, k, _), (outcome, offset)) in results if outcome isa String]
    sort!(failures)
    columns = [Matrix{Float64}(undef, n_moments, n_free) for _ in factors]
    schemes = Vector{Symbol}(undef, n_free)
    for a in eachindex(factors), k in 1:n_free
        column, scheme = _difference(base, get(results, (a, k, 1), nothing), get(results, (a, k, -1), nothing))
        columns[a][:, k] = column
        a == 1 && (schemes[k] = scheme)
    end
    jacobian_matrix = columns[1]
    sensitivity = fill(NaN, n_free)
    for k in 1:n_free
        # Near a narrow interval every factor can collapse to the same points.
        alternatives = [a for a in 2:length(factors) if perturbed[a, k] != perturbed[1, k]]
        isempty(alternatives) && continue
        reference = jacobian_matrix[:, k] ./ scales
        sensitivity[k] = maximum(alternatives) do a
            alternative = columns[a][:, k] ./ scales
            norm(alternative .- reference) / max(norm(reference), norm(alternative), floatmin())
        end
    end
    steps = [maximum(abs, perturbed[1, k] .- θ[free[k]]) for k in 1:n_free]
    return MomentJacobian(jacobian_matrix, θ, free, labels, lower, upper, steps, schemes, sensitivity, failures)
end

# Weighted, scaled Jacobian A = L D⁻¹ J with L'L = W and D = diag(scales):
# the derivative of the criterion residuals r, so that the loss is r'r.
function _criterion_jacobian(jac::MomentJacobian, objective::MomentObjective)
    size(jac.jacobian, 1) == length(objective.target) ||
        throw(ArgumentError("the Jacobian and the objective have different numbers of moments"))
    all(isfinite, jac.jacobian) ||
        throw(ArgumentError("the Jacobian has failed columns; see its failures"))
    return _multiply(objective.factor, jac.jacobian ./ objective.scales)
end

function _column_scales(criterion_jacobian, parameter_scales, jac::MomentJacobian)
    if parameter_scales === :column
        return [n > 0 ? 1 / n : 1.0 for n in norm.(eachcol(criterion_jacobian))]
    elseif parameter_scales === :raw
        return ones(size(criterion_jacobian, 2))
    end
    scales = _broadcast_vector(parameter_scales, 1.0, length(jac.parameters), "parameter_scales")[jac.free]
    (all(isfinite, scales) && all(>(0), scales)) ||
        throw(ArgumentError("parameter_scales must be :column, :raw, or positive finite values"))
    return scales
end

_numerical_rank(values, rank_tol) = isempty(values) || values[1] == 0 ? 0 : count(>(rank_tol * values[1]), values)

"""
    IdentificationReport

Singular value decomposition of the weighted, scaled moment Jacobian
`L D⁻¹ J C`, where `L'L = W` are the criterion weights, `D = diag(scales)`, and
`C = diag(parameter_scales)` normalizes the parameters. Column `k` of
`directions` is a parameter combination in normalized units (the physical
displacement is `parameter_scales .* directions[:, k]`) that moves the criterion
residuals by `singular_values[k]` per unit. Small singular values mark weakly
identified combinations; `rank` and `condition` summarize them.
"""
struct IdentificationReport
    names::Vector{String}
    parameter_scales::Vector{Float64}
    singular_values::Vector{Float64}
    directions::Matrix{Float64}
    rank::Int
    condition::Float64
end

"""
    identification(jac, objective; parameter_scales=:column, rank_tol=sqrt(eps())) -> IdentificationReport

Local identification diagnostics at the point where `jac` was evaluated. Moments
are normalized as in the criterion of `objective` (its `scales` and `weights`).
Parameter normalization is explicit because singular values depend on units:

- `:column` (default) scales each parameter so its column has unit norm, which
  makes the diagnostics invariant to parameter units and isolates collinearity
  in how moments respond (Belsley, Kuh & Welsch). Roundoff is normalized too, so
  a parameter that barely moves any moment shows up as a very large
  `parameter_scales` entry (the inverse column norm) and an unstable `sensitivity`
  in the Jacobian rather than as a small singular value.
- `:raw` keeps original parameter units.
- a vector over all parameters gives the physical size of one unit of each.

The rank counts singular values above `rank_tol` times the largest; `rank_tol`
should be no smaller than the relative accuracy of the Jacobian, which for
simulated moments is far coarser than the default. These are local diagnostics: a full-rank Jacobian does not establish global identification
and a poor fit or an unconverged search can mimic weak identification.
"""
function identification(jac::MomentJacobian, objective::MomentObjective;
                        parameter_scales=:column, rank_tol::Real=sqrt(eps()))
    criterion_jacobian = _criterion_jacobian(jac, objective)
    scales = _column_scales(criterion_jacobian, parameter_scales, jac)
    n_free = length(scales)
    decomposition = svd(criterion_jacobian .* scales'; full=true)
    values = [decomposition.S; zeros(n_free - length(decomposition.S))]
    directions = Matrix(decomposition.V)
    # Fix the arbitrary sign: the largest loading of each direction is positive.
    for column in eachcol(directions)
        column .*= sign(column[argmax(abs.(column))])
    end
    rank = _numerical_rank(values, rank_tol)
    condition = values[end] > 0 ? values[1] / values[end] : Inf
    return IdentificationReport(jac.names, scales, values, directions, rank, condition)
end

"""
    ParameterCovariance

Sandwich covariance of the free-parameter `estimate`. `covariance` and
`standard_errors` are `NaN` when the weighted Jacobian is rank deficient.
`near_bound[k]` marks estimates within `bound_se` standard errors of a finite
bound (or on it). `regular` is true only with full rank, stable derivatives, and
no flagged bound; `issues` explains any failure.
"""
struct ParameterCovariance
    names::Vector{String}
    estimate::Vector{Float64}
    covariance::Matrix{Float64}
    standard_errors::Vector{Float64}
    rank::Int
    near_bound::Vector{Bool}
    regular::Bool
    issues::Vector{String}
end

"""
    parameter_covariance(jac, objective; covariance, nobs, kwargs...) -> ParameterCovariance

Sandwich covariance of a minimum-distance estimate,

    V = (G'WG)⁻¹ G'W S W G (G'WG)⁻¹,   G = D⁻¹ J,   S = D⁻¹ (Σ / nobs) D⁻¹,

where `J` is the moment Jacobian, `W` and `D = diag(scales)` come from
`objective`, and `Σ = covariance` is the asymptotic covariance of
`√nobs (m̂ - m(θ₀))` in raw moment units, so `Σ / nobs` is the covariance of the
moment errors. It must include simulation noise: with simulated moments from
`R` independent samples the size of the data, a common choice is `(1 + 1/R)`
times the data covariance. Pass the covariance of the moment errors itself with
`nobs=1`. With `W = S⁻¹` this reduces to `(G'S⁻¹G)⁻¹`.

The estimate is not regular, and [`wald_intervals`](@ref) refuses it, when:
- the weighted Jacobian has rank below the number of parameters (with columns
  scaled to unit norm, so it does not depend on parameter units; `rank_tol`);
- a Jacobian column changes by more than `sensitivity_tol` across step sizes;
- an estimate lies within `bound_se=2` standard errors of a finite bound, where
  the normal approximation fails. The bounds are those of `jac`: compute it with
  the true parameter space, not an arbitrary search box. An estimate at the tail
  cutoff of an infinite bound is reported by [`search_summary`](@ref) instead.
"""
function parameter_covariance(jac::MomentJacobian, objective::MomentObjective;
                              covariance::AbstractMatrix{<:Real}, nobs::Real,
                              rank_tol::Real=sqrt(eps()), sensitivity_tol::Real=0.1, bound_se::Real=2)
    n_moments = length(objective.target)
    sigma = Matrix{Float64}(covariance)
    (size(sigma) == (n_moments, n_moments) && all(isfinite, sigma)) ||
        throw(ArgumentError("covariance must be a finite $n_moments × $n_moments matrix"))
    isapprox(sigma, sigma'; rtol=1e-10, atol=1e-12) || throw(ArgumentError("covariance must be symmetric"))
    eigenvalues = eigvals(Symmetric(sigma))
    eigenvalues[1] < -1e-10 * max(1.0, eigenvalues[end]) &&
        throw(ArgumentError("covariance must be positive semidefinite"))
    (isfinite(nobs) && nobs > 0) || throw(ArgumentError("nobs must be positive and finite"))
    (isfinite(bound_se) && bound_se >= 0) || throw(ArgumentError("bound_se must be nonnegative"))

    criterion_jacobian = _criterion_jacobian(jac, objective)
    scales = _column_scales(criterion_jacobian, :column, jac)
    n_free = length(scales)
    decomposition = svd(criterion_jacobian .* scales')
    rank = _numerical_rank(decomposition.S, rank_tol)
    issues = String[]
    if rank < n_free
        covariance_matrix = fill(NaN, n_free, n_free)
        push!(issues, "weighted moment Jacobian has rank $rank < $n_free parameters; see identification")
    else
        # In criterion-residual units the moment-error covariance is L S L'.
        scaled = (sigma ./ nobs) ./ (objective.scales .* objective.scales')
        residual_covariance = _multiply(objective.factor, _multiply(objective.factor, scaled)')
        # Pseudoinverse of the column-scaled Jacobian: (A'A)⁻¹A' with A = L G C.
        projection = decomposition.V * (decomposition.U ./ decomposition.S')'
        normalized = projection * residual_covariance * projection'
        covariance_matrix = scales .* (normalized .+ normalized') ./ 2 .* scales'
    end
    for k in findall(>(sensitivity_tol), jac.sensitivity)
        push!(issues, "the derivative with respect to $(jac.names[k]) changes by $(_short(jac.sensitivity[k])) " *
                      "across step sizes")
    end
    standard_errors = sqrt.(max.(diag(covariance_matrix), 0.0))
    estimate = jac.parameters[jac.free]
    distance = min.(estimate .- jac.lower, jac.upper .- estimate)
    near_bound = [isnan(se) ? d == 0 : d <= bound_se * se for (d, se) in zip(distance, standard_errors)]
    for k in findall(near_bound)
        push!(issues, distance[k] == 0 ? "$(jac.names[k]) is on a bound" :
                      "$(jac.names[k]) is within $(bound_se) standard errors of a bound")
    end
    return ParameterCovariance(jac.names, estimate, covariance_matrix, standard_errors, rank, near_bound,
                               isempty(issues), issues)
end

"""
    wald_intervals(result::ParameterCovariance; z=1.959963984540054) -> Matrix

Symmetric Wald intervals `estimate ± z * standard_errors` as rows `[lower upper]`
(the default `z` gives 95% coverage). Throws unless `result.regular`, because
rank deficiency, unstable derivatives, or a bound invalidates the normal approximation.
"""
function wald_intervals(result::ParameterCovariance; z::Real=1.959963984540054)
    result.regular || throw(ArgumentError("Wald intervals are not valid here: " * join(result.issues, "; ")))
    return [result.estimate .- z .* result.standard_errors result.estimate .+ z .* result.standard_errors]
end

# --- Display --------------------------------------------------------------------

function _loadings(names, direction; threshold=0.1)
    order = sortperm(abs.(direction); rev=true)
    terms = [string(direction[i] >= 0 ? "+" : "-", _short(abs(direction[i])), " ", names[i])
             for i in order if abs(direction[i]) >= threshold]
    return join(terms, " ")
end

function Base.show(io::IO, ::MIME"text/plain", jac::MomentJacobian)
    print(io, "MomentJacobian: ", size(jac.jacobian, 1), " moments × ", length(jac.free), " free parameters")
    rows = [["parameter", "value", "step", "scheme", "step sensitivity"]]
    for k in eachindex(jac.free)
        push!(rows, [jac.names[k], _short(jac.parameters[jac.free[k]]), _short(jac.steps[k]),
                     string(jac.schemes[k]), _short(jac.sensitivity[k])])
    end
    _print_rows(io, rows; left=(1, 4))
    isempty(jac.failures) || print(io, "\n  failed perturbations:\n    ", join(jac.failures, "\n    "))
end

function Base.show(io::IO, ::MIME"text/plain", report::IdentificationReport)
    print(io, "IdentificationReport: rank ", report.rank, " of ", length(report.names),
          ", condition number ", _short(report.condition))
    rows = [["singular value", "direction (normalized units)"]]
    for k in eachindex(report.singular_values)
        push!(rows, [_short(report.singular_values[k]), _loadings(report.names, report.directions[:, k])])
    end
    _print_rows(io, rows; left=(1, 2))
    print(io, "\n  parameter scales: ", join(("$name $(_short(c))" for (name, c) in
                                              zip(report.names, report.parameter_scales)), ", "))
end

function Base.show(io::IO, ::MIME"text/plain", result::ParameterCovariance)
    print(io, "ParameterCovariance: rank ", result.rank, " of ", length(result.names), ", ",
          result.regular ? "regular" : "not regular")
    rows = [["parameter", "estimate", "std. error", "near bound"]]
    for k in eachindex(result.names)
        push!(rows, [result.names[k], _short(result.estimate[k]), _short(result.standard_errors[k]),
                     result.near_bound[k] ? "yes" : ""])
    end
    _print_rows(io, rows)
    isempty(result.issues) || print(io, "\n  issues:\n    ", join(result.issues, "\n    "))
end

Base.show(io::IO, jac::MomentJacobian) =
    print(io, "MomentJacobian(", size(jac.jacobian, 1), " moments × ", length(jac.free), " parameters)")
Base.show(io::IO, report::IdentificationReport) =
    print(io, "IdentificationReport(rank ", report.rank, " of ", length(report.names), ")")
Base.show(io::IO, result::ParameterCovariance) =
    print(io, "ParameterCovariance(rank ", result.rank, " of ", length(result.names), ", ",
          result.regular ? "regular" : "not regular", ")")
