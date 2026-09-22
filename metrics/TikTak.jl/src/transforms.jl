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
    n = length(lower)
    n == 0 && throw(ArgumentError("bounds must contain at least one parameter"))
    if any(isnan, lower) || any(isnan, upper) || any(lower .> upper) ||
       any(==(Inf), lower) || any(==(-Inf), upper)
        throw(ArgumentError("invalid parameter bounds"))
    end
    (isfinite(tail) && 0 < tail < 0.5) || throw(ArgumentError("tail must lie strictly between 0 and 0.5"))
    s = _broadcast_vector(scale, 1.0, n, "scale")
    (all(isfinite, s) && all(>(0), s)) || throw(ArgumentError("scale must be finite and positive"))
    l = _broadcast_vector(location, 0.0, n, "location")
    all(isfinite, l) || throw(ArgumentError("location must be finite"))
    free = findall(lower .!= upper)
    transform = BoxTransform(lower, upper, s, l, Float64(tail), free, length(free), n)
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
function to_parameters(t::BoxTransform, unit::AbstractVector{<:Real})
    length(unit) == t.dimension || throw(ArgumentError("unit point has invalid length"))
    all(isfinite, unit) || throw(ArgumentError("unit point has nonfinite coordinates"))
    (any(<(0), unit) || any(>(1), unit)) && throw(ArgumentError("unit point lies outside [0, 1]"))
    x = copy(t.lower)
    for (k, j) in enumerate(t.free)
        v = Float64(unit[k])
        lo, hi, s = t.lower[j], t.upper[j], t.scale[j]
        if isfinite(lo) && isfinite(hi)
            x[j] = (1 - v) * lo + v * hi
        elseif isfinite(lo)
            z = (1 - t.tail) * v
            x[j] = lo + s * z / (1 - z)
        elseif isfinite(hi)
            z = (1 - t.tail) * (1 - v)
            x[j] = hi - s * z / (1 - z)
        else
            z = t.tail + (1 - 2 * t.tail) * v
            x[j] = t.location[j] + s * tan(pi * (z - 0.5))
        end
    end
    return x
end

"""
    to_unit(transform, parameters) -> Vector{Float64}

Unit-box coordinates of the free entries of a physical parameter vector. Throws
if the parameters violate the bounds or lie beyond the finite tail cutoff of an
infinite bound.
"""
function to_unit(t::BoxTransform, parameters::AbstractVector{<:Real})
    length(parameters) == t.size || throw(ArgumentError("parameters must be a vector matching bounds"))
    x = Vector{Float64}(parameters)
    all(isfinite, x) || throw(ArgumentError("parameters must be a finite vector matching bounds"))
    (any(x .< t.lower) || any(x .> t.upper)) && throw(ArgumentError("parameters violate bounds"))
    u = Vector{Float64}(undef, t.dimension)
    for (k, j) in enumerate(t.free)
        lo, hi, s = t.lower[j], t.upper[j], t.scale[j]
        if isfinite(lo) && isfinite(hi)
            width = hi - lo
            # Avoid overflow for very wide intervals, but preserve tiny ones.
            u[k] = isfinite(width) ? (x[j] - lo) / width : (x[j] / 2 - lo / 2) / (hi / 2 - lo / 2)
        elseif isfinite(lo)
            d = (x[j] - lo) / s
            u[k] = (1 - 1 / (1 + d)) / (1 - t.tail)
        elseif isfinite(hi)
            d = (hi - x[j]) / s
            u[k] = 1 - (1 - 1 / (1 + d)) / (1 - t.tail)
        else
            z = 0.5 + atan((x[j] - t.location[j]) / s) / pi
            u[k] = (z - t.tail) / (1 - 2 * t.tail)
        end
    end
    (any(<(-1e-12), u) || any(>(1 + 1e-12), u)) &&
        throw(ArgumentError("parameters exceed the finite tail cutoff"))
    return clamp.(u, 0.0, 1.0)
end

_json_number(v::Float64) = isfinite(v) ? v : string(v)

function specification(t::BoxTransform)
    return Dict{String,Any}(
        "bounds" => [[_json_number(lo), _json_number(hi)] for (lo, hi) in zip(t.lower, t.upper)],
        "scale" => copy(t.scale), "location" => copy(t.location), "tail" => t.tail)
end
