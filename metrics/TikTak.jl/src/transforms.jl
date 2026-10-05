# Coordinate maps between economic parameters and the finite optimization box.

"""
    BoxTransform(bounds; scale=nothing, location=nothing, tail=1e-6)

Map free parameters to the unit box `[0, 1]^d`; parameters with equal lower and
upper bounds are fixed and removed from the optimization dimension.

`bounds` is a vector of `(lower, upper)` pairs (or an `n × 2` matrix). Finite
bounds use an affine map. One-sided infinite bounds use a rational map and
two-sided infinite bounds a tangent map; `tail` truncates only the infinite ends
so that model inputs stay finite. `scale` sets the unit of exploration on
unbounded dimensions and `location` centers fully unbounded dimensions, both in
physical parameter units. Use [`to_parameters`](@ref) and [`to_unit`](@ref) to
move between the two coordinate systems.
"""
struct BoxTransform
    lower::Vector{Float64}
    upper::Vector{Float64}
    scale::Vector{Float64}
    location::Vector{Float64}
    tail::Float64
    free::Vector{Int}
    dimension::Int
    size::Int
end

function _parse_bounds(bounds::AbstractMatrix)
    size(bounds, 2) == 2 || throw(ArgumentError("bounds must have shape (number of parameters, 2)"))
    return Float64.(bounds[:, 1]), Float64.(bounds[:, 2])
end

function _parse_bounds(bounds)
    lower, upper = Float64[], Float64[]
    for pair in bounds
        length(pair) == 2 || throw(ArgumentError("each bound must be a (lower, upper) pair"))
        lo, hi = pair
        push!(lower, Float64(lo))
        push!(upper, Float64(hi))
    end
    return lower, upper
end

function BoxTransform(bounds; scale=nothing, location=nothing, tail=1e-6)
    lower, upper = _parse_bounds(bounds)
    n_parameters = length(lower)
    n_parameters == 0 && throw(ArgumentError("bounds must contain at least one parameter"))
    if any(isnan, lower) || any(isnan, upper) || any(lower .> upper) ||
       any(==(Inf), lower) || any(==(-Inf), upper)
        throw(ArgumentError("invalid parameter bounds"))
    end
    (isfinite(tail) && 0 < tail < 0.5) || throw(ArgumentError("tail must lie strictly between 0 and 0.5"))
    scales = _broadcast_vector(scale, 1.0, n_parameters, "scale")
    (all(isfinite, scales) && all(>(0), scales)) || throw(ArgumentError("scale must be finite and positive"))
    locations = _broadcast_vector(location, 0.0, n_parameters, "location")
    all(isfinite, locations) || throw(ArgumentError("location must be finite"))
    free = findall(lower .!= upper)
    transform = BoxTransform(lower, upper, scales, locations, Float64(tail), free, length(free), n_parameters)
    # Fail before launching a model if user-specified scales overflow.
    if !(all(isfinite, to_parameters(transform, zeros(transform.dimension))) &&
         all(isfinite, to_parameters(transform, ones(transform.dimension))))
        throw(ArgumentError("transformed bounds overflow; reduce scale or increase tail"))
    end
    return transform
end

"""
    to_parameters(transform, unit) -> Vector{Float64}

Physical parameter vector (including fixed parameters) for a point of the unit box.
"""
function to_parameters(transform::BoxTransform, unit::AbstractVector{<:Real})
    length(unit) == transform.dimension || throw(ArgumentError("unit point has invalid length"))
    all(isfinite, unit) || throw(ArgumentError("unit point has nonfinite coordinates"))
    (any(<(0), unit) || any(>(1), unit)) && throw(ArgumentError("unit point lies outside [0, 1]"))
    parameters = copy(transform.lower)
    for (unit_index, parameter_index) in enumerate(transform.free)
        coordinate = Float64(unit[unit_index])
        lower, upper, scale = (
            transform.lower[parameter_index],
            transform.upper[parameter_index],
            transform.scale[parameter_index],
        )
        if isfinite(lower) && isfinite(upper)
            parameters[parameter_index] = (1 - coordinate) * lower + coordinate * upper
        elseif isfinite(lower)
            mapped_coordinate = (1 - transform.tail) * coordinate
            parameters[parameter_index] = lower + scale * mapped_coordinate / (1 - mapped_coordinate)
        elseif isfinite(upper)
            mapped_coordinate = (1 - transform.tail) * (1 - coordinate)
            parameters[parameter_index] = upper - scale * mapped_coordinate / (1 - mapped_coordinate)
        else
            mapped_coordinate = transform.tail + (1 - 2 * transform.tail) * coordinate
            parameters[parameter_index] = transform.location[parameter_index] +
                scale * tan(pi * (mapped_coordinate - 0.5))
        end
    end
    return parameters
end

"""
    to_unit(transform, parameters) -> Vector{Float64}

Unit-box coordinates of the free entries of a physical parameter vector. Throws
if the parameters violate the bounds or lie beyond the finite tail cutoff of an
infinite bound.
"""
function to_unit(transform::BoxTransform, parameters::AbstractVector{<:Real})
    length(parameters) == transform.size || throw(ArgumentError("parameters must be a vector matching bounds"))
    physical = Vector{Float64}(parameters)
    all(isfinite, physical) || throw(ArgumentError("parameters must be a finite vector matching bounds"))
    (any(physical .< transform.lower) || any(physical .> transform.upper)) &&
        throw(ArgumentError("parameters violate bounds"))
    unit = Vector{Float64}(undef, transform.dimension)
    for (unit_index, parameter_index) in enumerate(transform.free)
        lower, upper, scale = (
            transform.lower[parameter_index],
            transform.upper[parameter_index],
            transform.scale[parameter_index],
        )
        if isfinite(lower) && isfinite(upper)
            width = upper - lower
            # Avoid overflow for very wide intervals, but preserve tiny ones.
            unit[unit_index] = isfinite(width) ?
                (physical[parameter_index] - lower) / width :
                (physical[parameter_index] / 2 - lower / 2) / (upper / 2 - lower / 2)
        elseif isfinite(lower)
            distance = (physical[parameter_index] - lower) / scale
            unit[unit_index] = (1 - 1 / (1 + distance)) / (1 - transform.tail)
        elseif isfinite(upper)
            distance = (upper - physical[parameter_index]) / scale
            unit[unit_index] = 1 - (1 - 1 / (1 + distance)) / (1 - transform.tail)
        else
            mapped_coordinate = 0.5 +
                atan((physical[parameter_index] - transform.location[parameter_index]) / scale) / pi
            unit[unit_index] = (mapped_coordinate - transform.tail) / (1 - 2 * transform.tail)
        end
    end
    (any(<(-1e-12), unit) || any(>(1 + 1e-12), unit)) &&
        throw(ArgumentError("parameters exceed the finite tail cutoff"))
    return clamp.(unit, 0.0, 1.0)
end

_json_number(v::Float64) = isfinite(v) ? v : string(v)

function specification(transform::BoxTransform)
    return Dict{String,Any}(
        "bounds" => [[_json_number(lo), _json_number(hi)] for (lo, hi) in zip(transform.lower, transform.upper)],
        "scale" => copy(transform.scale),
        "location" => copy(transform.location),
        "tail" => transform.tail,
    )
end
