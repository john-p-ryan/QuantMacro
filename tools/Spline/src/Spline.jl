module Spline


using LinearAlgebra


export CubicSpline, CubicSplineInterpolation,
       LinearSpline, LinearSplineInterpolation,
       PchipSpline, PchipSplineInterpolation,
       HymanSpline, HymanSplineInterpolation,
       BilinearSpline, BilinearSplineInterpolation,
       evaluate_spline, evaluate_spline_derivative, evaluate_spline_antiderivative,
       evaluate_spline!, evaluate_spline_derivative!, evaluate_spline_antiderivative!,
       safe_spline, safe_pchip,
       make_grid, evaluate_spline_grid


# --- Shared Helpers ---

# Bounds check for the in-place `evaluate_spline!` family.
@inline function _check_out(results, new_x)
    length(results) == length(new_x) ||
        throw(DimensionMismatch("output vector must have the same length as new_x"))
end

# Index of the cell containing `v`, clamped to a valid interval index.
@inline _cell_index(knots, v) = clamp(searchsortedfirst(knots, v) - 1, 1, length(knots) - 1)

# Same, but a comparison-only binary search so it also works for dual numbers
# (ForwardDiff), which `searchsortedfirst` cannot be relied on to handle.
@inline function _cell_index_ad(knots, v)
    lo, hi = 1, length(knots)
    while hi - lo > 1
        mid = (lo + hi) >>> 1
        if knots[mid] <= v
            lo = mid
        else
            hi = mid
        end
    end
    return lo
end

# Cumulative integral of a piecewise cubic at the knots: I[i] = int_{x[1]}^{x[i]}.
function _cumulative_integral(x, a, b, c, d)
    n = length(x)
    I = zeros(Float64, n)
    for i in 2:n
        h = x[i] - x[i-1]
        I[i] = I[i-1] + a[i-1] * h + b[i-1] * h^2 / 2 + c[i-1] * h^3 / 3 + d[i-1] * h^4 / 4
    end
    return I
end

# Segment coefficients of the cubic Hermite interpolant with knot slopes `m`:
# on [x[i], x[i+1]] the cubic is y[i] + b*dx + c*dx^2 + d*dx^3.
function _hermite_coeffs(h, delta, m)
    T = promote_type(eltype(h), eltype(delta), eltype(m))
    n = length(m)
    b = zeros(T, n - 1)
    c = zeros(T, n - 1)
    d = zeros(T, n - 1)
    for i = 1:n-1
        b[i] = m[i]
        c[i] = (3 * delta[i] - 2 * m[i] - m[i+1]) / h[i]
        d[i] = (m[i] + m[i+1] - 2 * delta[i]) / (h[i] * h[i])
    end
    return b, c, d
end

# Hyman (1983) filter. Clamps the knot slopes `m` so that every interval
# satisfies the Fritsch-Carlson sufficient condition for monotonicity,
# 0 <= m[i]/delta[i] <= 3 and 0 <= m[i+1]/delta[i] <= 3. Knots at which the
# secant slopes change sign (local extrema) or vanish (flat segments) get a
# zero slope; slopes already inside the region are left untouched.
function _hyman_filter!(m, delta)
    n = length(m)
    for i = 1:n
        # Secants on either side of the knot; the endpoints see one interval only.
        dl = delta[max(i - 1, 1)]
        dr = delta[min(i, n - 1)]
        if dl * dr > 0
            s = sign(dr)
            m[i] = s * clamp(s * m[i], zero(m[i]), 3 * min(abs(dl), abs(dr)))
        else
            m[i] = zero(m[i])
        end
    end
    return m
end

# Tridiagonal system for the interior second derivatives of a cubic spline.
function _cubic_diagonals(h, bc_type::String, ::Type{T}) where {T}
    n = length(h) + 1
    n_eq = n - 2
    dl = zeros(T, max(n_eq - 1, 0))
    dv = zeros(T, n_eq)
    du = zeros(T, max(n_eq - 1, 0))
    for i = 1:n_eq
        if i == 1  # First row: equation for c[2]
            if bc_type == "not-a-knot"
                dv[i] = (h[1] + h[2]) * (h[1] + 2*h[2]) / h[2]
                n_eq > 1 && (du[i] = h[2] - h[1]^2 / h[2])
            else  # natural or clamped
                dv[i] = 2 * (h[1] + h[2])
                n_eq > 1 && (du[i] = h[2])
            end
        elseif i == n_eq  # Last row: equation for c[n-1]
            if bc_type == "not-a-knot"
                dl[i-1] = h[n-2] - h[n-1]^2 / h[n-2]
                dv[i] = (h[n-2] + h[n-1]) * (h[n-1] + 2*h[n-2]) / h[n-2]
            else  # natural or clamped
                dl[i-1] = h[n-2]
                dv[i] = 2 * (h[n-2] + h[n-1])
            end
        else  # Interior rows: equation for c[i+1]
            dl[i-1] = h[i]
            dv[i] = 2 * (h[i] + h[i+1])
            du[i] = h[i+1]
        end
    end
    return Tridiagonal(dl, dv, du)
end

# Cached LU factorizations of that system. The matrix depends only on the knots
# and the boundary condition, so refitting new y-values on a fixed grid costs
# only the O(n) solve.
const _CUBIC_LU = Dict{Tuple{Vector{Float64},String},Factorization{Float64}}()
const _CUBIC_LU_LOCK = ReentrantLock()

function _cubic_factorization(x::Vector{Float64}, h::Vector{Float64}, bc_type::String)
    lock(_CUBIC_LU_LOCK) do
        key = (x, bc_type)
        haskey(_CUBIC_LU, key) && return _CUBIC_LU[key]
        length(_CUBIC_LU) >= 64 && empty!(_CUBIC_LU)  # keep the cache bounded
        F = lu(_cubic_diagonals(h, bc_type, Float64))
        _CUBIC_LU[(copy(x), bc_type)] = F
        return F
    end
end


# --- Cubic Spline Implementation ---

# Piecewise cubics stored as per-interval (a, b, c, d) with cubic extrapolation.
# `CubicSplineInterpolation` and `HymanSplineInterpolation` share the
# `evaluate_spline*` methods below through this supertype.
abstract type AbstractCubicSplineInterpolation end

# Define the Spline object to store spline information
struct CubicSplineInterpolation <: AbstractCubicSplineInterpolation
    x::Vector{Float64}
    y::Vector{Float64}
    a::Vector{Float64} # a[i] = y[i]
    b::Vector{Float64} # Coefficients for (x-x[i])
    c::Vector{Float64} # Coefficients for (x-x[i])^2
    d::Vector{Float64} # Coefficients for (x-x[i])^3
    I::Vector{Float64} # Cumulative integral at each knot (for antiderivatives)
    bc_type::String
    extrapolate::Bool
end

"""
    CubicSpline(x, y; bc_type="not-a-knot", extrapolate=true)

Constructs a CubicSplineInterpolation object from given x and y data.

# Arguments
- `x`: Array of x-coordinates of data points. Must be strictly increasing.
- `y`: Array of y-coordinates of data points. Must be the same length as x.
- `bc_type`: Boundary condition type. Can be "natural", "clamped", or "not-a-knot".
               Defaults to "natural". "clamped" is not yet fully implemented and will behave like natural.
- `extrapolate`: Boolean indicating whether to extrapolate for x values outside the
                 range of the input x data. Defaults to true.

# Returns
- A CubicSplineInterpolation object.
"""
function CubicSpline(x::Vector{Float64}, y::Vector{Float64}; bc_type::String="not-a-knot", extrapolate::Bool=true)
    n = length(x)
    if n != length(y)
        throw(DimensionMismatch("x and y must have the same length"))
    end
    if n <= 2
        throw(ArgumentError("At least 3 data points are required for cubic spline interpolation"))
    end
    if !issorted(x)
        throw(ArgumentError("x must be strictly increasing"))
    end
    bc_type_lower = lowercase(bc_type)
    if bc_type_lower ∉ ["natural", "clamped", "not-a-knot"]
        @warn "Boundary condition type '$bc_type' not recognized, defaulting to 'not-a-knot'."
        bc_type_lower = "not-a-knot"
    end

    h = diff(x)
    if any(h .<= 0)
        throw(ArgumentError("x must be strictly increasing"))
    end

    a = y  # a[i] = y[i]

    # Right-hand side of the tridiagonal system for the second derivatives
    n_eq = n - 2  # Number of equations
    B = zeros(Float64, n_eq)
    for i = 1:n_eq
        B[i] = 3 * ((y[i + 2] - y[i + 1]) / h[i + 1] - (y[i + 1] - y[i]) / h[i])
    end

    # Solve the tridiagonal system for c_interior. The factorization only
    # depends on x and bc_type, so it is cached and reused across refits.
    c_interior = _cubic_factorization(x, h, bc_type_lower) \ B

    c = zeros(Float64, n)
    c[2:n - 1] .= c_interior

    # Calculate c[1] and c[n] based on the not-a-knot conditions
    if bc_type_lower == "not-a-knot"
        c[1] = ((h[1] + h[2]) / h[2]) * c[2] - (h[1] / h[2]) * c[3]
        c[n] = ((h[n - 2] + h[n - 1]) / h[n - 2]) * c[n - 1] - (h[n - 1] / h[n - 2]) * c[n - 2]
    elseif bc_type_lower == "natural" || bc_type_lower == "clamped"
        c[1] = 0.0
        c[n] = 0.0
    end

    # Calculate b and d coefficients
    b = zeros(Float64, n - 1)
    d = zeros(Float64, n - 1)
    for i = 1:n - 1
        b[i] = (a[i + 1] - a[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])
    end

    I = _cumulative_integral(x, a, b, c, d)
    return CubicSplineInterpolation(x, y, a, b, c, d, I, bc_type_lower, extrapolate)
end



"""
    evaluate_spline(spline::CubicSplineInterpolation, new_x::Vector{Float64})
    evaluate_spline(spline::HymanSplineInterpolation, new_x::Vector{Float64})

Evaluates the cubic spline interpolation at new x values.

# Arguments
- `spline`: A CubicSplineInterpolation or HymanSplineInterpolation object.
- `new_x`: Array of new x-coordinates at which to evaluate the spline.

# Returns
- Array of interpolated y-values at the new x-coordinates.
"""
function evaluate_spline!(results::Vector{Float64}, spline::AbstractCubicSplineInterpolation, new_x::Vector{Float64})
    _check_out(results, new_x)
    n = length(spline.x)

    for (i, x_val) in enumerate(new_x)
        if !spline.extrapolate && (x_val < spline.x[1] || x_val > spline.x[end])
            results[i] = NaN
            continue
        end
        # Cubic extrapolation reuses the first/last interval's coefficients.
        idx = x_val < spline.x[1] ? 1 : (x_val > spline.x[end] ? n - 1 : _cell_index(spline.x, x_val))
        dx = x_val - spline.x[idx]
        results[i] = spline.a[idx] + spline.b[idx] * dx + spline.c[idx] * dx^2 + spline.d[idx] * dx^3
    end
    return results
end

evaluate_spline(spline::AbstractCubicSplineInterpolation, new_x::Vector{Float64}) =
    evaluate_spline!(zeros(Float64, length(new_x)), spline, new_x)



"""
    evaluate_spline_derivative(spline::CubicSplineInterpolation, new_x::Vector{Float64})
    evaluate_spline_derivative(spline::HymanSplineInterpolation, new_x::Vector{Float64})

Evaluates the derivative of the cubic spline interpolation at new x values,
consistent with cubic extrapolation when extrapolate=true.

# Arguments
- `spline`: A CubicSplineInterpolation or HymanSplineInterpolation object.
- `new_x`: Array of new x-coordinates at which to evaluate the derivative of the spline.

# Returns
- Array of derivative values at the new x-coordinates.
"""
function evaluate_spline_derivative!(results::Vector{Float64}, spline::AbstractCubicSplineInterpolation, new_x::Vector{Float64})
    _check_out(results, new_x)
    n = length(spline.x)

    for (i, x_val) in enumerate(new_x)
        if !spline.extrapolate && (x_val < spline.x[1] || x_val > spline.x[end])
            results[i] = NaN
            continue
        end
        idx = x_val < spline.x[1] ? 1 : (x_val > spline.x[end] ? n - 1 : _cell_index(spline.x, x_val))
        dx = x_val - spline.x[idx]
        results[i] = spline.b[idx] + 2 * spline.c[idx] * dx + 3 * spline.d[idx] * dx^2
    end
    return results
end

evaluate_spline_derivative(spline::AbstractCubicSplineInterpolation, new_x::Vector{Float64}) =
    evaluate_spline_derivative!(zeros(Float64, length(new_x)), spline, new_x)



"""
    evaluate_spline_antiderivative(spline::CubicSplineInterpolation, new_x::Vector{Float64}; C::Float64=0.0)
    evaluate_spline_antiderivative(spline::HymanSplineInterpolation, new_x::Vector{Float64}; C::Float64=0.0)

Evaluates the antiderivative (indefinite integral) of the cubic spline.

# Arguments
- `spline`: A `CubicSplineInterpolation` or `HymanSplineInterpolation` object.
- `new_x`:  A vector of x-values at which to evaluate the antiderivative.
- `C`: The constant of integration. Defaults to the negative of the antiderivative evaluated at spline.x[1].

# Returns
- A vector of the antiderivative values at the corresponding `new_x` points.

# Notes
The antiderivative is calculated interval by interval. The constant of integration
`C` is added to all results. The default value for `C` is chosen such that the
antiderivative is zero at the *first* knot point (`spline.x[1]`). Extrapolation
behavior is controlled by the `spline.extrapolate` setting.

"""
function evaluate_spline_antiderivative!(results::Vector{Float64}, spline::AbstractCubicSplineInterpolation, new_x::Vector{Float64}; C::Float64=NaN)
    _check_out(results, new_x)
    n = length(spline.x)

    # Cumulative integrals at the knots (spline.I) were computed once at
    # construction; they act as the integration constants for each segment.
    isnan(C) && (C = 0.0)  # Default: antiderivative is 0 at the first knot.

    for (i, x_val) in enumerate(new_x)
        if !spline.extrapolate && (x_val < spline.x[1] || x_val > spline.x[end])
            results[i] = NaN
            continue
        end
        idx = x_val < spline.x[1] ? 1 : (x_val > spline.x[end] ? n - 1 : _cell_index(spline.x, x_val))
        dx = x_val - spline.x[idx]
        results[i] = C + spline.I[idx] + spline.a[idx] * dx + spline.b[idx] * dx^2 / 2 +
                     spline.c[idx] * dx^3 / 3 + spline.d[idx] * dx^4 / 4
    end
    return results
end

evaluate_spline_antiderivative(spline::AbstractCubicSplineInterpolation, new_x::Vector{Float64}; C::Float64=NaN) =
    evaluate_spline_antiderivative!(zeros(Float64, length(new_x)), spline, new_x; C=C)



# --- AD-safe interpolants (ForwardDiff dual numbers) ---
#
# Build the coefficients once and reuse them across queries:
#     itp = safe_spline(x, y);  itp(xq)
# `safe_spline(x, y, x_new)` is the one-shot form for a scalar or a vector.

# Not-a-knot cubic coefficients, generic in the element type.
function _spline_coeffs(x, y)
    n = length(x)
    n == length(y) || throw(DimensionMismatch("x and y must have the same length"))
    n < 3 && throw(ArgumentError("At least 3 data points are required for cubic spline interpolation"))

    T = promote_type(eltype(x), eltype(y))
    h = diff(x)

    # Solve the tridiagonal system for the interior second derivatives
    B = T[3 * ((y[i+2] - y[i+1]) / h[i+1] - (y[i+1] - y[i]) / h[i]) for i = 1:n-2]
    c = zeros(T, n)
    c[2:n-1] .= _cubic_diagonals(h, "not-a-knot", T) \ B

    # Not-a-knot boundary conditions for the first and last points
    c[1] = ((h[1] + h[2]) / h[2]) * c[2] - (h[1] / h[2]) * c[3]
    c[n] = ((h[n-2] + h[n-1]) / h[n-2]) * c[n-1] - (h[n-1] / h[n-2]) * c[n-2]

    b = zeros(T, n - 1)
    d = zeros(T, n - 1)
    for i = 1:n-1
        b[i] = (y[i+1] - y[i]) / h[i] - h[i] * (c[i+1] + 2 * c[i]) / 3
        d[i] = (c[i+1] - c[i]) / (3 * h[i])
    end
    return b, c, d
end

# Scalar evaluation with cubic extrapolation outside the knot range.
function _spline_value(x, y, b, c, d, xq)
    n = length(x)
    i = xq <= x[1] ? 1 : (xq >= x[end] ? n - 1 : _cell_index_ad(x, xq))
    dx = xq - x[i]
    return y[i] + b[i] * dx + c[i] * dx^2 + d[i] * dx^3
end

"""
    safe_spline(x, y) -> callable
    safe_spline(x, y, x_new)

Cubic (not-a-knot) interpolation that works with dual numbers for automatic
differentiation via ForwardDiff. The two-argument form returns a callable that
holds the fitted coefficients, so repeated scalar queries do not refit.
"""
function safe_spline(x, y)
    b, c, d = _spline_coeffs(x, y)
    return xq -> _spline_value(x, y, b, c, d, xq)
end

function safe_spline(x, y, x_new)
    b, c, d = _spline_coeffs(x, y)
    x_new isa AbstractArray || return _spline_value(x, y, b, c, d, x_new)
    T = promote_type(eltype(x), eltype(y), eltype(x_new))
    return T[_spline_value(x, y, b, c, d, xq) for xq in x_new]
end



# --- Hyman-filtered Cubic Spline Implementation ---

struct HymanSplineInterpolation <: AbstractCubicSplineInterpolation
    x::Vector{Float64}
    y::Vector{Float64}
    m::Vector{Float64}  # Filtered slopes at each knot
    a::Vector{Float64}  # a[i] = y[i]
    b::Vector{Float64}  # Coefficients for (x-x[i])
    c::Vector{Float64}  # Coefficients for (x-x[i])^2
    d::Vector{Float64}  # Coefficients for (x-x[i])^3
    I::Vector{Float64}  # Cumulative integral at each knot (for antiderivatives)
    bc_type::String
    extrapolate::Bool
end

"""
    HymanSpline(x, y; bc_type="not-a-knot", extrapolate=true)

Constructs a HymanSplineInterpolation object: a cubic spline whose knot slopes
have been passed through the Hyman (1983) monotonicity filter.

The knot derivatives of the ordinary cubic spline (`CubicSpline` with the same
`bc_type`) are clamped so that on every interval `[x[i], x[i+1]]` the
Fritsch-Carlson sufficient condition for monotonicity holds:
`0 <= m[i]/delta[i] <= 3` and `0 <= m[i+1]/delta[i] <= 3`, where `delta[i]` is
the secant slope. Knots where the data has a local extremum get a zero slope.
The filtered slopes then define a piecewise cubic Hermite interpolant, which is
C¹ and monotone wherever the data is monotone.

On every interval whose two knot slopes pass the filter unchanged, the result
coincides with the cubic spline. In particular, if the cubic spline already
satisfies the condition at every knot, the two are identical. The condition is
sufficient but not necessary, so a monotone cubic spline can still be modified
if a knot slope exceeds three times the neighbouring secant slope.

# Arguments
- `x`: Array of x-coordinates of data points. Must be strictly increasing.
- `y`: Array of y-coordinates of data points. Must be the same length as x.
- `bc_type`: Boundary condition of the underlying cubic spline. Can be "natural",
             "clamped", or "not-a-knot". Defaults to "not-a-knot".
- `extrapolate`: Boolean indicating whether to extrapolate for x values outside the
                 range of the input x data. Defaults to true. Extrapolation is
                 cubic, using the end intervals, as for `CubicSpline`.

# Returns
- A HymanSplineInterpolation object.
"""
function HymanSpline(x::Vector{Float64}, y::Vector{Float64}; bc_type::String="not-a-knot", extrapolate::Bool=true)
    spl = CubicSpline(x, y; bc_type=bc_type, extrapolate=extrapolate)
    n = length(x)
    h = diff(x)
    delta = diff(y) ./ h

    # Knot slopes of the cubic spline: b[i] at the left end of each interval,
    # and the derivative of the last interval at x[n].
    m = zeros(Float64, n)
    m[1:n-1] .= spl.b
    m[n] = spl.b[n-1] + 2 * spl.c[n-1] * h[n-1] + 3 * spl.d[n-1] * h[n-1]^2

    _hyman_filter!(m, delta)

    a = y
    b, c, d = _hermite_coeffs(h, delta, m)
    I = _cumulative_integral(x, a, b, c, d)
    return HymanSplineInterpolation(x, y, m, a, b, c, d, I, spl.bc_type, extrapolate)
end

# Evaluation, derivative, and antiderivative are shared with CubicSpline via
# AbstractCubicSplineInterpolation.



# --- Linear Spline Implementation ---

struct LinearSplineInterpolation
    x::Vector{Float64}
    y::Vector{Float64}
    slopes::Vector{Float64}
    I::Vector{Float64}  # Cumulative integral at each knot (for antiderivatives)
    extrapolate::Bool
end

"""
    LinearSpline(x, y; extrapolate=true)

Constructs a LinearSplineInterpolation object from given x and y data.

# Arguments
- `x`: Array of x-coordinates of data points. Must be strictly increasing.
- `y`: Array of y-coordinates of data points. Must be the same length as x.
- `extrapolate`: Boolean indicating whether to extrapolate for x values outside the
                 range of the input x data. Defaults to true.

# Returns
- A LinearSplineInterpolation object.
"""
function LinearSpline(x::Vector{Float64}, y::Vector{Float64}; extrapolate::Bool=true)
    n = length(x)
    if n != length(y)
        throw(DimensionMismatch("x and y must have the same length"))
    end
    if n < 2
        throw(ArgumentError("At least 2 data points are required for linear spline interpolation"))
    end
    if !issorted(x)
        throw(ArgumentError("x must be strictly increasing"))
    end

    slopes = diff(y) ./ diff(x)
    zs = zero(slopes)
    I = _cumulative_integral(x, y, slopes, zs, zs)
    return LinearSplineInterpolation(x, y, slopes, I, extrapolate)
end

"""
    evaluate_spline(spline::LinearSplineInterpolation, new_x::Vector{Float64})

Evaluates the linear spline interpolation at new x values.

# Arguments
- `spline`: A LinearSplineInterpolation object.
- `new_x`: Array of new x-coordinates at which to evaluate the spline.

# Returns
- Array of interpolated y-values at the new x-coordinates.
"""
function evaluate_spline!(results::Vector{Float64}, spline::LinearSplineInterpolation, new_x::Vector{Float64})
    _check_out(results, new_x)
    n = length(spline.x)

    for (i, x_val) in enumerate(new_x)
        if !spline.extrapolate && (x_val < spline.x[1] || x_val > spline.x[end])
            results[i] = NaN
            continue
        end
        # Extrapolation continues the first/last segment.
        idx = x_val < spline.x[1] ? 1 : (x_val > spline.x[end] ? n - 1 : _cell_index(spline.x, x_val))
        results[i] = spline.y[idx] + spline.slopes[idx] * (x_val - spline.x[idx])
    end
    return results
end

evaluate_spline(spline::LinearSplineInterpolation, new_x::Vector{Float64}) =
    evaluate_spline!(zeros(Float64, length(new_x)), spline, new_x)

"""
    evaluate_spline_derivative(spline::LinearSplineInterpolation, new_x::Vector{Float64})

Evaluates the derivative of the linear spline.  The derivative is piecewise
constant.

# Arguments
- `spline`: A LinearSplineInterpolation object.
- `new_x`: Array of new x-coordinates at which to evaluate the derivative.

# Returns
- Array of derivative values at the new x-coordinates.
"""
function evaluate_spline_derivative!(results::Vector{Float64}, spline::LinearSplineInterpolation, new_x::Vector{Float64})
    _check_out(results, new_x)
    n = length(spline.x)

    for (i, x_val) in enumerate(new_x)
        if !spline.extrapolate && (x_val < spline.x[1] || x_val > spline.x[end])
            results[i] = NaN  # Indicate extrapolation is not allowed
            continue
        end
        idx = x_val < spline.x[1] ? 1 : (x_val > spline.x[end] ? n - 1 : _cell_index(spline.x, x_val))
        results[i] = spline.slopes[idx]
    end
    return results
end

evaluate_spline_derivative(spline::LinearSplineInterpolation, new_x::Vector{Float64}) =
    evaluate_spline_derivative!(zeros(Float64, length(new_x)), spline, new_x)


"""
    evaluate_spline_antiderivative(spline::LinearSplineInterpolation, new_x::Vector{Float64}; C::Float64=NaN)

Evaluates the antiderivative (indefinite integral) of the linear spline.

# Arguments
- `spline`: A `LinearSplineInterpolation` object.
- `new_x`: A vector of x-values at which to evaluate the antiderivative.
- `C`: The constant of integration.  Defaults such that the
      antiderivative is zero at the *first* knot (`spline.x[1]`).

# Returns
- A vector of the antiderivative values.
"""
function evaluate_spline_antiderivative!(results::Vector{Float64}, spline::LinearSplineInterpolation, new_x::Vector{Float64}; C::Float64=NaN)
    _check_out(results, new_x)
    n = length(spline.x)

    # Cumulative integrals at the knots (spline.I) come from the constructor.
    isnan(C) && (C = 0.0)  # Default: antiderivative is 0 at the first knot

    for (i, x_val) in enumerate(new_x)
        if !spline.extrapolate && (x_val < spline.x[1] || x_val > spline.x[end])
            results[i] = NaN
            continue
        end
        if x_val > spline.x[end]
            # Extrapolation to the right starts from the last knot.
            dx = x_val - spline.x[n]
            results[i] = C + spline.I[n] + spline.y[n] * dx + 0.5 * spline.slopes[n-1] * dx^2
        else
            idx = x_val < spline.x[1] ? 1 : _cell_index(spline.x, x_val)
            dx = x_val - spline.x[idx]
            results[i] = C + spline.I[idx] + spline.y[idx] * dx + 0.5 * spline.slopes[idx] * dx^2
        end
    end
    return results
end

evaluate_spline_antiderivative(spline::LinearSplineInterpolation, new_x::Vector{Float64}; C::Float64=NaN) =
    evaluate_spline_antiderivative!(zeros(Float64, length(new_x)), spline, new_x; C=C)




# --- Pchip Spline Implementation ---

# Define the Spline object to store spline information
struct PchipSplineInterpolation
    x::Vector{Float64}
    y::Vector{Float64}
    m::Vector{Float64}  # Slopes at each point
    a::Vector{Float64}  # a[i] = y[i]
    b::Vector{Float64}  # Coefficients for (x-x[i])
    c::Vector{Float64}  # Coefficients for (x-x[i])^2
    d::Vector{Float64}  # Coefficients for (x-x[i])^3
    I::Vector{Float64}  # Cumulative integral at each knot (for antiderivatives)
    extrapolate::Bool
end

"""
    PchipSpline(x, y; extrapolate=true)

Constructs a PchipSplineInterpolation object from given x and y data.

# Arguments
- `x`: Array of x-coordinates of data points. Must be strictly increasing.
- `y`: Array of y-coordinates of data points. Must be the same length as x.
- `extrapolate`: Boolean indicating whether to extrapolate for x values outside the
                 range of the input x data. Defaults to true.

# Returns
- A PchipSplineInterpolation object.
"""
function PchipSpline(x::Vector{Float64}, y::Vector{Float64}; extrapolate::Bool=true)
    n = length(x)
    if n != length(y)
        throw(DimensionMismatch("x and y must have the same length"))
    end
    if n < 2
        throw(ArgumentError("At least 2 data points are required for PCHIP interpolation"))
    end
    if !issorted(x)
        throw(ArgumentError("x must be strictly increasing"))
    end

    h = diff(x)
    delta = diff(y) ./ h

    # Compute slopes (m) using Fritsch-Carlson method
    m = zeros(Float64, n)

    # Interior points
    for i = 2:n-1
        if sign(delta[i-1]) != sign(delta[i])
            m[i] = 0.0
        else
            w1 = 2 * h[i] + h[i-1]
            w2 = h[i] + 2 * h[i-1]
            m[i] = (w1 + w2) / (w1 / delta[i-1] + w2 / delta[i])
        end
    end

    # Endpoint slopes (special handling to ensure monotonicity and shape preservation)
    m[1] = ((2 * h[1] + h[2]) * delta[1] - h[1] * delta[2]) / (h[1] + h[2])
    if sign(m[1]) != sign(delta[1])
      m[1] = 0.0
    elseif sign(delta[1]) != sign(delta[2]) && abs(m[1]) > abs(3*delta[1])
        m[1] = 3*delta[1]
    end
    
    m[n] = ((2 * h[n-1] + h[n-2]) * delta[n-1] - h[n-1] * delta[n-2]) / (h[n-1] + h[n-2])
    if sign(m[n]) != sign(delta[n-1])
        m[n] = 0.0
    elseif sign(delta[n-1]) != sign(delta[n-2]) && abs(m[n]) > abs(3 * delta[n-1])
      m[n] = 3*delta[n-1]
    end


    # Calculate coefficients
    a = y
    b, c, d = _hermite_coeffs(h, delta, m)

    I = _cumulative_integral(x, a, b, c, d)
    return PchipSplineInterpolation(x, y, m, a, b, c, d, I, extrapolate)
end

"""
    evaluate_spline(spline::PchipSplineInterpolation, new_x::Vector{Float64})

Evaluates the PCHIP spline interpolation at new x values.

# Arguments
- `spline`: A PchipSplineInterpolation object.
- `new_x`: Array of new x-coordinates at which to evaluate the spline.

# Returns
- Array of interpolated y-values at the new x-coordinates.
"""
function evaluate_spline!(results::Vector{Float64}, spline::PchipSplineInterpolation, new_x::Vector{Float64})
    _check_out(results, new_x)
    n = length(spline.x)

    for (i, x_val) in enumerate(new_x)
        if x_val < spline.x[1] || x_val > spline.x[end]
            # Extrapolation (linear, using the slope at the endpoint)
            if spline.extrapolate
                j = x_val < spline.x[1] ? 1 : n
                results[i] = spline.a[j] + spline.m[j] * (x_val - spline.x[j])
            else
                results[i] = NaN
            end
        else
            idx = _cell_index(spline.x, x_val)
            dx = x_val - spline.x[idx]
            results[i] = spline.a[idx] + spline.b[idx] * dx + spline.c[idx] * dx^2 + spline.d[idx] * dx^3
        end
    end
    return results
end

evaluate_spline(spline::PchipSplineInterpolation, new_x::Vector{Float64}) =
    evaluate_spline!(zeros(Float64, length(new_x)), spline, new_x)


"""
    evaluate_spline_derivative(spline::PchipSplineInterpolation, new_x::Vector{Float64})

Evaluates the derivative of the PCHIP spline interpolation at new x values.

# Arguments
- `spline`: A PchipSplineInterpolation object.
- `new_x`: Array of new x-coordinates at which to evaluate the derivative.

# Returns
- Array of derivative values at the new x-coordinates.
"""
function evaluate_spline_derivative!(results::Vector{Float64}, spline::PchipSplineInterpolation, new_x::Vector{Float64})
    _check_out(results, new_x)
    n = length(spline.x)

    for (i, x_val) in enumerate(new_x)
        if x_val < spline.x[1] || x_val > spline.x[end]
            # Extrapolation (constant, using the slope at the endpoint)
            results[i] = spline.extrapolate ? spline.m[x_val < spline.x[1] ? 1 : n] : NaN
        else
            idx = _cell_index(spline.x, x_val)
            dx = x_val - spline.x[idx]
            results[i] = spline.b[idx] + 2 * spline.c[idx] * dx + 3 * spline.d[idx] * dx^2
        end
    end
    return results
end

evaluate_spline_derivative(spline::PchipSplineInterpolation, new_x::Vector{Float64}) =
    evaluate_spline_derivative!(zeros(Float64, length(new_x)), spline, new_x)


"""
    evaluate_spline_antiderivative(spline::PchipSplineInterpolation, new_x::Vector{Float64}; C::Float64=NaN)

Evaluates the antiderivative (indefinite integral) of the PCHIP spline.

# Arguments
- `spline`: A `PchipSplineInterpolation` object.
- `new_x`: A vector of x-values at which to evaluate the antiderivative.
- `C`:  Constant of integration. Defaults to make the antiderivative zero at x[1].

# Returns
- A vector of the antiderivative values at the corresponding `new_x` points.

# Notes
Like with the cubic spline, the integral is computed piecewise, using a
pre-calculated cumulative integral at the knots.
"""
function evaluate_spline_antiderivative!(results::Vector{Float64}, spline::PchipSplineInterpolation, new_x::Vector{Float64}; C::Float64=NaN)
    _check_out(results, new_x)
    n = length(spline.x)

    # Cumulative integrals at the knots (spline.I) come from the constructor.
    isnan(C) && (C = 0.0)  # Default: antiderivative is 0 at the first knot

    for (i, x_val) in enumerate(new_x)
        if x_val < spline.x[1] || x_val > spline.x[end]
            if spline.extrapolate
                # Linear extrapolation, using the slope at the endpoint.
                j = x_val < spline.x[1] ? 1 : n
                dx = x_val - spline.x[j]
                results[i] = C + spline.I[j] + spline.y[j] * dx + 0.5 * spline.m[j] * dx^2
            else
                results[i] = NaN  # Outside the domain, return NaN.
            end
        else
            idx = _cell_index(spline.x, x_val)
            dx = x_val - spline.x[idx]
            results[i] = C + spline.I[idx] + spline.a[idx] * dx + spline.b[idx] * dx^2 / 2 +
                         spline.c[idx] * dx^3 / 3 + spline.d[idx] * dx^4 / 4
        end
    end
    return results
end

evaluate_spline_antiderivative(spline::PchipSplineInterpolation, new_x::Vector{Float64}; C::Float64=NaN) =
    evaluate_spline_antiderivative!(zeros(Float64, length(new_x)), spline, new_x; C=C)


# --- AD-safe PCHIP (ForwardDiff dual numbers) ---

# Fritsch-Carlson slopes and segment coefficients, generic in the element type.
function _pchip_coeffs(x, y)
    n = length(x)
    n == length(y) || throw(DimensionMismatch("x and y must have the same length"))
    n < 2 && throw(ArgumentError("At least 2 data points are required for PCHIP interpolation"))

    T = promote_type(eltype(x), eltype(y))
    h = diff(x)
    delta = diff(y) ./ h
    m = zeros(T, n)

    # Interior points
    for i = 2:n-1
        if delta[i-1] * delta[i] <= 0  # Different signs or one is zero
            m[i] = 0.0
        else
            w1 = 2 * h[i] + h[i-1]
            w2 = h[i] + 2 * h[i-1]
            m[i] = (w1 + w2) / (w1 / delta[i-1] + w2 / delta[i])
        end
    end

    # Endpoint slopes with monotonicity preservation
    if n > 2
        m[1] = ((2 * h[1] + h[2]) * delta[1] - h[1] * delta[2]) / (h[1] + h[2])
        if delta[1] * m[1] <= 0  # Different signs or slope is zero
            m[1] = 0.0
        elseif (delta[1] * delta[2] <= 0) && (abs(m[1]) > abs(3 * delta[1]))
            m[1] = 3 * delta[1]
        end

        m[n] = ((2 * h[n-1] + h[n-2]) * delta[n-1] - h[n-1] * delta[n-2]) / (h[n-1] + h[n-2])
        if delta[n-1] * m[n] <= 0  # Different signs or slope is zero
            m[n] = 0.0
        elseif (delta[n-1] * delta[n-2] <= 0) && (abs(m[n]) > abs(3 * delta[n-1]))
            m[n] = 3 * delta[n-1]
        end
    else
        # Special case for n=2: use simple secant slopes
        m[1] = delta[1]
        m[n] = delta[1]
    end

    b, c, d = _hermite_coeffs(h, delta, m)
    return m, b, c, d
end

# Scalar evaluation with linear extrapolation outside the knot range.
function _pchip_value(x, y, m, b, c, d, xq)
    n = length(x)
    if xq <= x[1]
        return y[1] + m[1] * (xq - x[1])
    elseif xq >= x[end]
        return y[n] + m[n] * (xq - x[n])
    end
    i = _cell_index_ad(x, xq)
    dx = xq - x[i]
    return y[i] + b[i] * dx + c[i] * dx^2 + d[i] * dx^3
end

"""
    safe_pchip(x, y) -> callable
    safe_pchip(x, y, x_new)

PCHIP interpolation that works with dual numbers for automatic differentiation
via ForwardDiff. The two-argument form returns a callable that holds the fitted
slopes and coefficients, so repeated scalar queries do not refit the spline.
"""
function safe_pchip(x, y)
    m, b, c, d = _pchip_coeffs(x, y)
    return xq -> _pchip_value(x, y, m, b, c, d, xq)
end

function safe_pchip(x, y, x_new)
    m, b, c, d = _pchip_coeffs(x, y)
    x_new isa AbstractArray || return _pchip_value(x, y, m, b, c, d, x_new)
    T = promote_type(eltype(x), eltype(y), eltype(x_new))
    return T[_pchip_value(x, y, m, b, c, d, xq) for xq in x_new]
end


# --- Bilinear Spline Implementation ---

struct BilinearSplineInterpolation
    x::Vector{Float64}  # x grid points
    y::Vector{Float64}  # y grid points
    z::Matrix{Float64}  # z values at grid points z[i,j] = f(x[i], y[j])
    # Precomputed coefficients for each cell
    # For a cell defined by (x[i], y[j]), (x[i+1], y[j+1]), we store:
    a::Matrix{Float64}  # a[i,j] = z[i,j] (value at lower left corner)
    b::Matrix{Float64}  # b[i,j] = coefficient for (x-x[i])
    c::Matrix{Float64}  # c[i,j] = coefficient for (y-y[j])
    d::Matrix{Float64}  # d[i,j] = coefficient for (x-x[i])*(y-y[j])
    extrapolate::Bool   # Whether to extrapolate for points outside the grid
    bc_type::String     # Boundary condition type: "linear" or "constant"
end

"""
    BilinearSpline(x, y, z; extrapolate=true, bc_type="linear")

Constructs a BilinearSplineInterpolation object from given x, y grid and z values.

# Arguments
- `x`: Array of x-coordinates forming a grid. Must be strictly increasing.
- `y`: Array of y-coordinates forming a grid. Must be strictly increasing.
- `z`: Matrix of z-values at the grid points, where z[i,j] corresponds to f(x[i], y[j]).
       Must have dimensions length(x) × length(y).
- `extrapolate`: Boolean indicating whether to extrapolate for points outside the
                 range of the input grid. Defaults to true.
- `bc_type`: String indicating the boundary condition type for extrapolation.
             Options are:
             - "linear": Linear extrapolation (continues the interpolating function)
             - "constant": Constant extrapolation (uses the nearest edge value)
             Defaults to "linear".

# Returns
- A BilinearSplineInterpolation object.
"""
function BilinearSpline(x::Vector{Float64}, y::Vector{Float64}, z::Matrix{Float64}; 
                        extrapolate::Bool=true, bc_type::String="linear")
    nx = length(x)
    ny = length(y)
    
    if size(z) != (nx, ny)
        throw(DimensionMismatch("z matrix dimensions must match the length of x and y: expected size $(nx) × $(ny), got $(size(z))"))
    end
    
    if nx < 2 || ny < 2
        throw(ArgumentError("At least 2 data points in each dimension are required for bilinear interpolation"))
    end
    
    if !issorted(x) || !issorted(y)
        throw(ArgumentError("x and y must be strictly increasing"))
    end
    
    if !(bc_type in ["linear", "constant"])
        throw(ArgumentError("bc_type must be one of 'linear' or 'constant'"))
    end
    
    # Precompute coefficients for each cell in the grid
    a = zeros(Float64, nx-1, ny-1)
    b = zeros(Float64, nx-1, ny-1)
    c = zeros(Float64, nx-1, ny-1)
    d = zeros(Float64, nx-1, ny-1)
    
    for i in 1:(nx-1)
        for j in 1:(ny-1)
            # Cell corner values
            z11 = z[i, j]
            z21 = z[i+1, j]
            z12 = z[i, j+1]
            z22 = z[i+1, j+1]
            
            # Cell dimensions
            dx = x[i+1] - x[i]
            dy = y[j+1] - y[j]
            
            # Bilinear interpolation coefficients
            a[i, j] = z11
            b[i, j] = (z21 - z11) / dx
            c[i, j] = (z12 - z11) / dy
            d[i, j] = (z22 - z21 - z12 + z11) / (dx * dy)
        end
    end
    
    return BilinearSplineInterpolation(x, y, z, a, b, c, d, extrapolate, bc_type)
end

"""
    evaluate_spline(spline::BilinearSplineInterpolation, new_x::Vector{Float64}, new_y::Vector{Float64})

Evaluates the bilinear spline interpolation at new (x,y) coordinates.

# Arguments
- `spline`: A BilinearSplineInterpolation object.
- `new_x`: Array of new x-coordinates at which to evaluate the spline.
- `new_y`: Array of new y-coordinates at which to evaluate the spline.
               Must have the same length as new_x.

# Returns
- Array of interpolated z-values at the new coordinates.
"""
function evaluate_spline!(results::Vector{Float64}, spline::BilinearSplineInterpolation,
                          new_x::Vector{Float64}, new_y::Vector{Float64})
    if length(new_x) != length(new_y)
        throw(DimensionMismatch("new_x and new_y must have the same length"))
    end
    _check_out(results, new_x)

    for i in eachindex(new_x)
        results[i] = _bilinear_value(spline, new_x[i], new_y[i])
    end
    return results
end

evaluate_spline(spline::BilinearSplineInterpolation, new_x::Vector{Float64}, new_y::Vector{Float64}) =
    evaluate_spline!(zeros(Float64, length(new_x)), spline, new_x, new_y)

# Scalar evaluation honouring the extrapolation setting and boundary condition.
@inline function _bilinear_value(spline::BilinearSplineInterpolation, x_val::Float64, y_val::Float64)
    outside = x_val < spline.x[1] || x_val > spline.x[end] ||
              y_val < spline.y[1] || y_val > spline.y[end]
    if outside
        spline.extrapolate || return NaN
        if spline.bc_type == "constant"
            # Use closest point on the boundary for constant extrapolation
            x_val = clamp(x_val, spline.x[1], spline.x[end])
            y_val = clamp(y_val, spline.y[1], spline.y[end])
        end
    end
    return evaluate_point(spline, x_val, y_val)
end

"""
    evaluate_point(spline::BilinearSplineInterpolation, x_val::Float64, y_val::Float64)

Helper function that evaluates the bilinear spline at a single point (x_val, y_val).

# Arguments
- `spline`: A BilinearSplineInterpolation object.
- `x_val`: x-coordinate at which to evaluate the spline.
- `y_val`: y-coordinate at which to evaluate the spline.

# Returns
- Interpolated z-value at the given coordinates.
"""
@inline function evaluate_point(spline::BilinearSplineInterpolation, x_val::Float64, y_val::Float64)
    # Find grid cell containing the point
    x_idx = _cell_index(spline.x, x_val)
    y_idx = _cell_index(spline.y, y_val)

    # Use precomputed coefficients for bilinear interpolation
    # f(x,y) = a + b*(x-x1) + c*(y-y1) + d*(x-x1)*(y-y1)
    return _cell_value(spline, x_idx, y_idx, x_val - spline.x[x_idx], y_val - spline.y[y_idx])
end

@inline _cell_value(spline::BilinearSplineInterpolation, i::Int, j::Int, dx::Float64, dy::Float64) =
    spline.a[i, j] + spline.b[i, j] * dx + spline.c[i, j] * dy + spline.d[i, j] * dx * dy




"""
    evaluate_spline_grid(spline::BilinearSplineInterpolation, grid_x::Vector{Float64}, grid_y::Vector{Float64})

Evaluates the bilinear spline on a 2D grid of points. Useful for visualization.

# Arguments
- `spline`: A BilinearSplineInterpolation object.
- `grid_x`: Array of x-coordinates forming a grid.
- `grid_y`: Array of y-coordinates forming a grid.

# Returns
- A matrix of z-values with dimensions length(grid_x) × length(grid_y),
  where result[i,j] is the interpolated value at (grid_x[i], grid_y[j]).
"""
function evaluate_spline_grid(spline::BilinearSplineInterpolation, grid_x::Vector{Float64}, grid_y::Vector{Float64})
    nx = length(grid_x)
    ny = length(grid_y)
    result = zeros(Float64, nx, ny)

    # Locate every query coordinate once instead of per cell.
    clamped = spline.extrapolate && spline.bc_type == "constant"
    ix, dx, inx = _locate(spline.x, grid_x, clamped)
    iy, dy, iny = _locate(spline.y, grid_y, clamped)

    # Column-major: the first matrix index runs innermost.
    for j in 1:ny
        for i in 1:nx
            result[i, j] = (spline.extrapolate || (inx[i] && iny[j])) ?
                _cell_value(spline, ix[i], iy[j], dx[i], dy[j]) : NaN
        end
    end

    return result
end

# Cell index, offset inside the cell, and an inside-the-range flag for each query.
function _locate(knots::Vector{Float64}, q::Vector{Float64}, clamped::Bool)
    idx = Vector{Int}(undef, length(q))
    off = Vector{Float64}(undef, length(q))
    inside = Vector{Bool}(undef, length(q))
    for (k, v) in enumerate(q)
        inside[k] = knots[1] <= v <= knots[end]
        clamped && (v = clamp(v, knots[1], knots[end]))
        i = _cell_index(knots, v)
        idx[k] = i
        off[k] = v - knots[i]
    end
    return idx, off, inside
end







# --- Grid Tools ---

"""
    make_grid(x_min, x_max, n, density=1.0)

Constructs a grid of `n` points between `x_min` and `x_max` with a given density. Higher density
values will concentrate more points near the lower bound `x_min`.

# Arguments
- `x_min`: Lower bound of the grid.
- `x_max`: Upper bound of the grid.
- `n`: Number of grid points.
- `density`: Density of grid points. Defaults to 1.0.

# Returns
- A vector of grid points.
"""
function make_grid(x_min::Float64, x_max::Float64, n::Int; density::Float64=1.0)
    x = range(0, 1, length=n)
    x = x .^ density
    x = x_min .+ x * (x_max - x_min)
    return x
end

end # module Spline


# --- Example Usage 1d ---

#=
using Spline

x = collect(range(0.3, 5, 15))
f(x) = log(x)
y = f.(x)

# construct spline object and evaluate
spl = CubicSpline(x, y) #
x_new = collect(range(0.05, 8, 100))
y_spl = evaluate_spline(spl, x_new)
y_true = f.(x_new)

# evaluate derivatives of spline
f_prime(x) = 1 / x
y_prime_true = f_prime.(x_new)
y_prime_spl = evaluate_spline_derivative(spl, x_new)

# evaluate antiderivatives of the spline with constant of integration
f_int(x) = x * log(x) - x
y_int_true = f_int.(x_new)
C = f_int(x[end]) - evaluate_spline_antiderivative(spl, [x[end]])[1]
y_int_spl = evaluate_spline_antiderivative(spl, x_new) .+ C


# plot results
using Plots
plot(x_new, y_true, label="True", lw=2)
plot!(x_new, y_spl, label="Interpolated", lw=2)
scatter!(x, y, label="Spline Points")

# plot derivatives
plot(x_new, y_prime_true, label="True Derivative", lw=2)
plot!(x_new, y_prime_spl, label="Interpolated Derivative", lw=2)
scatter!(x, f_prime.(x), label="Spline Points")

# plot antiderivative
plot(x_new, y_int_true, label="True Antiderivative", lw=2)
plot!(x_new, y_int_spl, label="Interpolated Antiderivative", lw=2)
scatter!(x, f_int.(x), label="Spline Points")
=#
