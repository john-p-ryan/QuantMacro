# Model-independent evaluation types and the method-of-moments criterion.

"""
    ModelEvaluationError(message)

Expected economic or numerical infeasibility, for example no equilibrium exists
or an inner solver failed to converge. Objectives that throw it are recorded as
failed evaluations and receive an infinite objective value, so a failure can
never beat a legitimate large objective. Programming errors should not be
converted into this exception.
"""
struct ModelEvaluationError <: Exception
    msg::String
end
ModelEvaluationError() = ModelEvaluationError("model evaluation failed")
Base.showerror(io::IO, e::ModelEvaluationError) = print(io, e.msg)

"""
    Evaluation(value; moments=nothing, residuals=nothing)

Return this from an objective to preserve model diagnostics alongside the scalar
loss. `moments` and `residuals` are stored with every evaluation in the run
history.
"""
struct Evaluation
    value::Float64
    moments::Union{Nothing,Vector{Float64}}
    residuals::Union{Nothing,Vector{Float64}}
end

function Evaluation(value::Real; moments=nothing, residuals=nothing)
    return Evaluation(Float64(value), _optional_vector(moments, "moments"),
                      _optional_vector(residuals, "residuals"))
end

_optional_vector(::Nothing, name) = nothing
_optional_vector(v::AbstractVector{<:Real}, name) = Vector{Float64}(v)
_optional_vector(v, name) = throw(ArgumentError("Evaluation.$name must be a real vector"))

# Broadcast a scalar, `nothing`, or a vector to a length-`n` Float64 vector.
_broadcast_vector(::Nothing, default, n, name) = fill(Float64(default), n)
_broadcast_vector(x::Real, default, n, name) = fill(Float64(x), n)
function _broadcast_vector(x::AbstractVector{<:Real}, default, n, name)
    length(x) == n || throw(ArgumentError("$name must have one entry per element ($n)"))
    return Vector{Float64}(x)
end
_broadcast_vector(x, default, n, name) = throw(ArgumentError("$name must be a real scalar or vector"))

"""
    MomentObjective(model, target; weights=nothing, scales=nothing)

Wrap `model(θ) -> moments` as the minimum-distance criterion
`r'r` with `r = L * ((moments - target) ./ scales)` and `L'L = W`.

`scales` standardizes moment errors before weighting. `weights` is a
nonnegative diagonal vector or a symmetric positive-semidefinite matrix.
Identity weights and unit scales are the defaults. No covariance matrix is
estimated or inverted here. Use fixed simulation draws so that evaluations
are deterministic and cacheable. Calling the objective returns an
[`Evaluation`](@ref) carrying the moments and residuals.
"""
struct MomentObjective{M}
    model::M
    target::Vector{Float64}
    scales::Vector{Float64}
    weights::Union{Vector{Float64},Matrix{Float64}}
    factor::Union{Vector{Float64},Matrix{Float64}}
end

function MomentObjective(model, target::AbstractVector{<:Real}; weights=nothing, scales=nothing)
    targets = Vector{Float64}(target)
    (isempty(targets) || !all(isfinite, targets)) &&
        throw(ArgumentError("target must be a nonempty finite vector"))
    n_moments = length(targets)
    moment_scales = _broadcast_vector(scales, 1.0, n_moments, "scales")
    (all(isfinite, moment_scales) && all(>(0), moment_scales)) ||
        throw(ArgumentError("moment scales must be positive and finite"))
    if weights === nothing
        weight_values = ones(n_moments)
        factor = ones(n_moments)
    elseif weights isa AbstractVector
        weight_values = Vector{Float64}(weights)
        length(weight_values) == n_moments ||
            throw(ArgumentError("weights must be a diagonal vector or square matrix"))
        all(isfinite, weight_values) || throw(ArgumentError("weights must be finite"))
        any(<(0), weight_values) && throw(ArgumentError("diagonal weights cannot be negative"))
        factor = sqrt.(weight_values)
    elseif weights isa AbstractMatrix
        weight_values = Matrix{Float64}(weights)
        size(weight_values) == (n_moments, n_moments) ||
            throw(ArgumentError("weights must be a diagonal vector or square matrix"))
        all(isfinite, weight_values) || throw(ArgumentError("weights must be finite"))
        isapprox(weight_values, weight_values'; rtol=1e-10, atol=1e-12) ||
            throw(ArgumentError("weight matrix must be symmetric"))
        eigenvalues, eigenvectors = eigen(Symmetric((weight_values + weight_values') / 2))
        if minimum(eigenvalues) < -1e-12 * max(1.0, maximum(abs, eigenvalues))
            throw(ArgumentError("weight matrix must be positive semidefinite"))
        end
        factor = sqrt.(max.(eigenvalues, 0.0)) .* eigenvectors'   # factor' * factor == W
    else
        throw(ArgumentError("weights must be a diagonal vector or square matrix"))
    end
    return MomentObjective(model, targets, moment_scales, weight_values, factor)
end

function (objective::MomentObjective)(parameters)
    raw = objective.model(parameters)
    raw isa AbstractVector{<:Real} || throw(ArgumentError("model must return a real vector of moments"))
    moments = Vector{Float64}(raw)
    length(moments) == length(objective.target) ||
        throw(ArgumentError("model moments have a different length than target"))
    all(isfinite, moments) || throw(ModelEvaluationError("model returned nonfinite moments"))
    errors = (moments .- objective.target) ./ objective.scales
    residuals = objective.factor isa Vector ? objective.factor .* errors : objective.factor * errors
    return Evaluation(dot(residuals, residuals), moments, residuals)
end

function specification(objective::MomentObjective)
    weights = objective.weights
    return Dict{String,Any}(
        "target" => copy(objective.target),
        "weights" => weights isa Vector ? copy(weights) : [collect(row) for row in eachrow(weights)],
        "scales" => copy(objective.scales),
    )
end
