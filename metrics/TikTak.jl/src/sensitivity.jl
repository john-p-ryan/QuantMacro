# Local moment responses and the linearized estimator's response to target changes.

"""
    SensitivitySummary

Named local responses from [`sensitivity_summary`](@ref). Rows of `jacobian`
and `scaled_jacobian` are moments; columns are free parameters. Rows of
`target_sensitivity` and `scaled_target_sensitivity` are free parameters;
columns are original moments, including when the criterion uses full weights.

With `J = jacobian`, `D = diag(moment_scales)`, and
`C = diag(parameter_scales)`:
- `scaled_jacobian = D⁻¹ J C` describes moment responses in scale units.
- `target_sensitivity = K` maps a change in raw targets to a change in physical
  parameters: `Δθ ≈ K * Δtarget`.
- `scaled_target_sensitivity = C⁻¹ K D` maps target scale units to parameter
  scale units.

`free` indexes the original parameter vector. `rank` uses the weighted Jacobian
with columns normalized to unit norm, independently of the reporting scales.
Both target-sensitivity matrices are `NaN` when rank deficient, since parameter
responses are then not uniquely identified. `on_bound` flags binding bounds.
`regular` requires full rank, stable derivatives, and no binding bounds;
`issues` explains failures. This does not certify convergence or goodness of fit.
"""
struct SensitivitySummary
    parameter_names::Vector{String}
    moment_names::Vector{String}
    free::Vector{Int}
    parameter_scales::Vector{Float64}
    moment_scales::Vector{Float64}
    jacobian::Matrix{Float64}
    scaled_jacobian::Matrix{Float64}
    target_sensitivity::Matrix{Float64}
    scaled_target_sensitivity::Matrix{Float64}
    rank::Int
    on_bound::Vector{Bool}
    regular::Bool
    issues::Vector{String}
end

"""
    sensitivity_summary(jac, objective; moment_names=nothing,
                        parameter_scales=:raw, rank_tol=sqrt(eps()),
                        sensitivity_tol=0.1) -> SensitivitySummary

Summarize which moments respond to each free parameter and how the criterion's
weights affect the linearized parameter response to changes in moment targets.
No model evaluation or reoptimization is performed. Parameter names come from
`jac`; moment names default to `"m1"`, `"m2"`, ...

For `G = D⁻¹ J`, `D = diag(objective.scales)`, and criterion weights `W`,

    K = (G' W G)⁻¹ G' W D⁻¹,    Δθ ≈ K * Δtarget.

The positive sign is for an increase in the empirical target. An additive
increase in model moments holding the target fixed has the opposite sign.
The calculation uses a column-normalized SVD, without forming normal equations
or explicit inverses. Full weight matrices retain cross-moment effects in the
original moment coordinates. Compare reports using different objectives' weights
at the same Jacobian to examine weighting choices.

Reporting units are explicit: `parameter_scales=:raw` (default) uses one physical
parameter unit; a positive scalar or vector over *all* parameters sets the size
of one unit, or `:column` uses inverse weighted column norms as in
[`identification`](@ref). Moment units are always the objective's `scales`.
Fixed parameters are omitted. Reporting scales do not affect raw `K` or rank.

This is a local, unconstrained Gauss–Newton approximation with weights, scales,
and simulation draws held fixed. It is exact for a linear moment model at an
interior optimum, and gives the derivative of a nonlinear optimum at an exact
fit. With nonzero residuals, curvature terms in the first-order conditions can
change the actual derivative. Binding bounds and unstable finite differences
are flagged but retain the unconstrained calculation; deficient rank returns
`NaN` target sensitivities instead of a nonunique pseudoinverse solution.
Unlike [`parameter_covariance`](@ref), no covariance is supplied here, so
proximity to bounds in standard-error units is not assessed.
"""
function sensitivity_summary(jac::MomentJacobian, objective::MomentObjective;
                             moment_names=nothing, parameter_scales=:raw,
                             rank_tol::Real=sqrt(eps()), sensitivity_tol::Real=0.1)
    (isfinite(rank_tol) && 0 <= rank_tol < 1) ||
        throw(ArgumentError("rank_tol must be finite and in [0, 1)"))
    (isfinite(sensitivity_tol) && sensitivity_tol >= 0) ||
        throw(ArgumentError("sensitivity_tol must be nonnegative and finite"))
    criterion = _criterion_jacobian(jac, objective)
    n_moments, n_free = size(criterion)
    labels = _labels(moment_names, n_moments, "m", "moment_names")
    units = _column_scales(criterion, parameter_scales, jac)
    # Use the same unit-invariant rank convention as sandwich inference.
    column_scales = _column_scales(criterion, :column, jac)
    decomposition = svd(criterion .* column_scales')
    rank = _numerical_rank(decomposition.S, rank_tol)
    issues = String[]
    if rank < n_free
        response = fill(NaN, n_free, n_moments)
        push!(issues, "weighted moment Jacobian has rank $rank < $n_free parameters; see identification")
    else
        projection = decomposition.V * (decomposition.U ./ decomposition.S')'
        # C A⁺ L D⁻¹, where A = L D⁻¹ J C. Keep diagonal factors as vectors.
        weighted = objective.factor isa Vector ? projection .* objective.factor' :
                   projection * objective.factor
        response = column_scales .* weighted
        response ./= objective.scales'
    end
    for k in findall(>(sensitivity_tol), jac.sensitivity)
        push!(issues, "the derivative with respect to $(jac.names[k]) changes by $(_short(jac.sensitivity[k])) " *
                      "across step sizes")
    end
    estimate = jac.parameters[jac.free]
    on_bound = (estimate .== jac.lower) .| (estimate .== jac.upper)
    for k in findall(on_bound)
        push!(issues, "$(jac.names[k]) is on a bound; target sensitivity is unconstrained")
    end
    return SensitivitySummary(copy(jac.names), labels, copy(jac.free), units, copy(objective.scales),
                              copy(jac.jacobian), jac.jacobian ./ objective.scales .* units',
                              response, response ./ units .* objective.scales',
                              rank, on_bound, isempty(issues), issues)
end

function Base.show(io::IO, ::MIME"text/plain", s::SensitivitySummary)
    print(io, "SensitivitySummary: local linear approximation, rank ", s.rank, " of ",
          length(s.parameter_names), ", ", s.regular ? "regular" : "not regular")
    rows = [["parameter", "moment", "dm/dθ", "scaled response", "dθ/dtarget", "scaled sensitivity"]]
    for k in eachindex(s.parameter_names), i in eachindex(s.moment_names)
        push!(rows, [s.parameter_names[k], s.moment_names[i], _short(s.jacobian[i, k]),
                     _short(s.scaled_jacobian[i, k]), _short(s.target_sensitivity[k, i]),
                     _short(s.scaled_target_sensitivity[k, i])])
    end
    _print_rows(io, rows; left=(1, 2))
    print(io, "\n  parameter scales: ", join(_short.(s.parameter_scales), ", "),
          "\n  moment scales: ", join(_short.(s.moment_scales), ", "),
          "\n  positive target sensitivity means the parameter rises when that target increases")
    isempty(s.issues) || print(io, "\n  issues:\n    ", join(s.issues, "\n    "))
end

Base.show(io::IO, s::SensitivitySummary) =
    print(io, "SensitivitySummary(rank ", s.rank, " of ", length(s.parameter_names), ", ",
          length(s.moment_names), " moments)")
