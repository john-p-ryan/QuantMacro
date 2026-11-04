# Inverse-covariance weighting matrices with explicit, unit-free regularization.
#
# Regularization and diagnostics use the correlation matrix R of the moment-error
# covariance Ω, so they do not depend on moment units or on `MomentObjective.scales`.
# Shrinking R towards the identity is the same as shrinking Ω towards its diagonal.

"""
    InverseCovarianceWeights

Regularized inverse `Ω_λ⁻¹` of the moment-error covariance `Ω = Σ / nobs`, built by
[`inverse_covariance_weights`](@ref). `precision` is in raw moment units (a vector of
inverse variances when `diagonal`). Pass the result as `weights` to
[`MomentObjective`](@ref), which converts it to its scale units as `W = D Ω_λ⁻¹ D` with
`D = diag(scales)`, so the criterion is `(m - target)' Ω_λ⁻¹ (m - target)` whatever the
scales.

Diagnostics refer to the correlation matrix `R` of `Ω`: its `eigenvalues` in ascending
order with the moment combinations `directions` as columns, its `condition` number and
numerical `rank`. Directions combine standardized moments (moment errors divided by
their `standard_deviations`), and a small eigenvalue marks a combination with little
variance. `shrinkage` is the applied `λ` (1 for diagonal weights) and
`regularized_condition` the condition number of `R_λ = (1 - λ) R + λ I`. The weights are
the efficient inverse-covariance weights only when `shrinkage == 0`.
"""
struct InverseCovarianceWeights
    names::Vector{String}
    precision::Union{Vector{Float64},Matrix{Float64}}
    standard_deviations::Vector{Float64}
    eigenvalues::Vector{Float64}
    directions::Matrix{Float64}
    condition::Float64
    rank::Int
    shrinkage::Float64
    regularized_condition::Float64
    diagonal::Bool
end

# Smallest λ with cond((1 - λ) R + λ I) <= κ, given the ascending eigenvalues of R.
# The eigenvalues of R_λ are (1 - λ) μᵢ + λ, so the bound is linear in λ.
function _required_shrinkage(values, max_condition)
    excess = values[end] - max_condition * values[1]
    return excess <= 0 ? 0.0 : excess / (excess + max_condition - 1)
end

"""
    inverse_covariance_weights(covariance; nobs, simulation_covariance=nothing,
                               simulation_nobs=nothing, n_simulations=nothing,
                               diagonal=false, shrinkage=0, max_condition=nothing,
                               rank_tol=sqrt(eps()), moment_names=nothing) -> InverseCovarianceWeights

Weights from the inverse of the moment-error covariance `Ω = Σ / nobs + Σₛ / nₛ`. The
keywords follow the conventions of [`parameter_covariance`](@ref): `Σ = covariance` is
the asymptotic covariance of `√nobs (m̂ - m(θ₀))` in raw moment units, and
`simulation_covariance` with `simulation_nobs`, or `n_simulations`, add the simulation
noise in the model moments; omit them when `Σ` already includes it or the moments are
computed without simulation. Without regularization these are the efficient
minimum-distance weights, and the minimized criterion is Hansen's J statistic.

Regularization shrinks the moment correlations towards zero,

    Ω_λ = (1 - λ) Ω + λ Diagonal(Ω),   equivalently   R_λ = (1 - λ) R + λ I,

which does not depend on moment units:

- `shrinkage=λ` in `[0, 1]` fixes the shrinkage.
- `max_condition=κ >= 1` uses the smallest `λ` for which `cond(R_λ) <= κ`; with
  `shrinkage` as well, the larger `λ` is used.
- `diagonal=true` uses the inverse variances alone (`λ = 1`), stored as a vector so that
  [`fit_summary`](@ref) attributes the loss to individual moments.

Throws if a moment has zero variance, or if an eigenvalue of `R_λ` is at most
`rank_tol` times the largest, as when a moment is a linear combination of others; the
message names the degenerate combination of moments. The eigenvalues of an estimated
`R` are more dispersed than the true ones, so unregularized weights can put large weight
on noisy moment combinations and bias estimates in finite samples (Altonji & Segal 1996).
Regularized and diagonal weights trade efficiency for robustness; the sandwich formula
of [`parameter_covariance`](@ref) remains valid for them.
"""
function inverse_covariance_weights(covariance::AbstractMatrix{<:Real}; nobs::Real, simulation_covariance=nothing,
                                    simulation_nobs=nothing, n_simulations=nothing, diagonal::Bool=false,
                                    shrinkage::Real=0, max_condition=nothing, rank_tol::Real=sqrt(eps()),
                                    moment_names=nothing)
    (isfinite(shrinkage) && 0 <= shrinkage <= 1) || throw(ArgumentError("shrinkage must lie in [0, 1]"))
    (max_condition === nothing || (max_condition isa Real && isfinite(max_condition) && max_condition >= 1)) ||
        throw(ArgumentError("max_condition must be a finite number of at least 1"))
    (isfinite(rank_tol) && 0 <= rank_tol < 1) || throw(ArgumentError("rank_tol must be finite and in [0, 1)"))
    sampling, simulation = _error_covariances(size(covariance, 1); covariance=covariance, nobs=nobs,
                                              simulation_covariance=simulation_covariance,
                                              simulation_nobs=simulation_nobs, n_simulations=n_simulations)
    omega = simulation === nothing ? sampling : sampling .+ simulation
    labels = _labels(moment_names, size(omega, 1), "m", "moment_names")
    variances = diag(omega)
    degenerate = findall(<=(0), variances)
    isempty(degenerate) || throw(ArgumentError(
        "moments with zero variance cannot be weighted by inverse covariance: $(join(labels[degenerate], ", "))"))
    deviations = sqrt.(variances)
    decomposition = eigen(Symmetric(omega ./ (deviations .* deviations')))
    values, directions = decomposition.values, decomposition.vectors
    # Unlike the check on Σ, this tolerance does not depend on moment units.
    values[1] < -1e-10 * values[end] && throw(ArgumentError("covariance must be positive semidefinite"))
    values = max.(values, 0.0)
    # Fix the arbitrary sign: the largest loading of each direction is positive.
    for column in eachcol(directions)
        column .*= sign(column[argmax(abs.(column))])
    end
    condition = values[1] > 0 ? values[end] / values[1] : Inf
    rank = _numerical_rank(reverse(values), rank_tol)

    if diagonal
        return InverseCovarianceWeights(labels, 1 ./ variances, deviations, values, directions,
                                        condition, rank, 1.0, 1.0, true)
    end
    λ = max(Float64(shrinkage), max_condition === nothing ? 0.0 : _required_shrinkage(values, max_condition))
    regularized = (1 - λ) .* values .+ λ
    regularized_condition = regularized[1] > 0 ? regularized[end] / regularized[1] : Inf
    if regularized[1] <= rank_tol * regularized[end]
        after = λ == 0 ? "" : " after shrinkage $(_short(λ))"
        throw(ArgumentError("the moment correlation matrix is numerically singular " *
                            "(condition number $(_short(regularized_condition))$after); the standardized " *
                            "combination $(_loadings(labels, directions[:, 1])) has almost no variance. " *
                            "Pass shrinkage or max_condition, use diagonal=true, or drop redundant moments"))
    end
    inverse_correlation = (directions ./ regularized') * directions'
    precision = inverse_correlation ./ (deviations .* deviations')
    return InverseCovarianceWeights(labels, (precision .+ precision') ./ 2, deviations, values, directions,
                                    condition, rank, λ, regularized_condition, false)
end

function _scaled_weights(weights::InverseCovarianceWeights, scales)
    length(weights.names) == length(scales) || throw(ArgumentError(
        "inverse-covariance weights have $(length(weights.names)) moments but the target has $(length(scales))"))
    weights.diagonal && return scales .^ 2 .* weights.precision
    scaled = scales .* weights.precision .* scales'
    return (scaled .+ scaled') ./ 2
end

function Base.show(io::IO, ::MIME"text/plain", w::InverseCovarianceWeights)
    form = w.diagonal ? "diagonal (inverse variances)" :
           w.shrinkage == 0 ? "full inverse covariance" :
           "full, shrinkage $(_short(w.shrinkage)) towards the diagonal"
    print(io, "InverseCovarianceWeights: ", form, "\n  correlation matrix: rank ", w.rank, " of ",
          length(w.names), ", condition number ", _short(w.condition))
    w.diagonal || w.shrinkage == 0 || print(io, ", ", _short(w.regularized_condition), " after shrinkage")
    rows = [["eigenvalue", "moment combination"]]
    for k in eachindex(w.eigenvalues)
        push!(rows, [_short(w.eigenvalues[k]), _loadings(w.names, w.directions[:, k])])
    end
    _print_rows(io, rows; left=(1, 2))
end

Base.show(io::IO, w::InverseCovarianceWeights) =
    print(io, "InverseCovarianceWeights(", length(w.names), " moments, shrinkage ", _short(w.shrinkage), ")")
