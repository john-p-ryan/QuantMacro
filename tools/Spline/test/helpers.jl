# Shared data sets and numerical helpers for the Spline test suite.

# --- Data sets ---

# Non-uniform knots with a smooth function. The SciPy reference values below
# were generated on these with scipy 1.18.1 (CubicSpline, PchipInterpolator).
const REF_X = [0.0, 0.5, 1.5, 2.0, 3.5, 4.0]
const REF_Y = sin.(REF_X)
const REF_XQ = [0.25, 1.0, 3.0, -1.0, 5.0]  # three inside, two outside the knots

# SciPy CubicSpline(x, y, bc_type=..., extrapolate=True); antiderivative is
# zero at x[1], matching this package's default constant of integration.
const SCIPY_CUBIC = Dict(
    "not-a-knot" => (
        value     = [0.25068367115507362, 0.83558786543718422, 0.13435067435915357, -1.04467273694251350, -0.89883217233073698],
        deriv     = [0.9647670766996468, 0.5417334459648144, -0.9772716127076095, 0.9599031889754732, 0.5209753725809367],
        antideriv = [0.03170014231287407, 0.45888166997566604, 1.98162493641585535, 0.52857662714410913, 0.71772777668142251],
    ),
    "natural" => (
        value     = [0.24807220402565133, 0.83687789576893801, 0.12393997765940720, -0.82510012163161006, -1.67548711684272122],
        deriv     = [0.96999699017313912, 0.53909755158412509, -0.97542129359184693, 0.46843090676015287, -1.20307418499659802],
        antideriv = [0.03118318039328037, 0.45860593444190290, 1.97591031755504032, 0.45713371267473712, 0.45756255836825149],
    ),
)

# SciPy PchipInterpolator on REF data. Only the three interior query points are
# comparable: SciPy extrapolates with the end cubics, this package linearly.
const SCIPY_PCHIP = (
    value     = [0.26533908598104039, 0.82542991511432473, 0.12968325386285123],
    deriv     = [0.9878927390560692, 0.6031648669793851, -1.0324208959883383],
    antideriv = [0.03378137297621610, 0.45654136099662451, 1.99062751510094826],
    slopes    = [1.1057782869445909, 0.6957572200815683, 0.0, -0.2629523907424557, -0.8234812654228293, -0.8050347266265542],
)

# Secant sign changes at both ends: exercises the PCHIP endpoint clamps.
const PCHIP2_X = [0.0, 1.0, 2.0, 3.5, 4.0]
const PCHIP2_Y = [0.0, 0.1, -1.0, -1.0, 2.0]
const PCHIP2_XQ = [0.5, 1.5, 3.0, 3.75]
const SCIPY_PCHIP2 = (
    slopes = [0.30000000000000004, 0.0, 0.0, 0.0, 7.5],
    value  = [0.08750000000000002, -0.45000000000000007, -1.0, 0.03125],
)

# Monotone data on which an unconstrained cubic spline overshoots.
const STEP_X = collect(0.0:9.0)
const STEP_Y = [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0]

# RPN14 data set from Fritsch & Carlson (1980).
const RPN14_X = [7.99, 8.09, 8.19, 8.7, 9.2, 10.0, 12.0, 15.0, 20.0]
const RPN14_Y = [0.0, 2.76429e-5, 4.37498e-2, 0.169183, 0.469428, 0.943740, 0.998636, 0.999919, 0.999994]

const MONOTONE_DATA = ["step" => (STEP_X, STEP_Y), "RPN14" => (RPN14_X, RPN14_Y)]

# Data with duplicated knots, which every constructor documents as invalid.
const DUP_X = [0.0, 1.0, 1.0, 2.0]
const DUP_Y = [0.0, 1.0, 2.0, 3.0]

const GRIDS = [
    "uniform"     => collect(range(0.0, 3.0, 8)),
    "non-uniform" => [0.0, 0.1, 0.35, 0.8, 1.0, 1.9, 2.4, 3.0],
]
const BC_TYPES = ["not-a-knot", "natural", "clamped"]

# --- Test functions ---

f_smooth(x) = exp(-0.5x) * sin(2x) + 0.3x

# Cubic polynomial with its derivative and antiderivative.
p3(x)  = 1 - 2x + 0.5x^2 - 0.25x^3
dp3(x) = -2 + x - 0.75x^2
P3(x)  = x - x^2 + x^3 / 6 - x^4 / 16

# Linear function with its antiderivative.
p1(x) = 2 - 0.5x
P1(x) = 2x - 0.25x^2

# --- Numerical helpers ---

# Central finite difference of a vector evaluator `f(xs::Vector)` at `xs`.
fd_derivative(f, xs; h=1e-6) = (f(xs .+ h) .- f(xs .- h)) ./ (2h)

# Query points strictly inside each knot interval (quarter, half, and
# three-quarter points) so finite differences never straddle a knot.
function interior_points(x)
    pts = Float64[]
    for i in 1:length(x)-1
        h = x[i+1] - x[i]
        append!(pts, x[i] .+ h .* (0.25, 0.5, 0.75))
    end
    return pts
end

# Points beyond both ends of the knot range.
outside_points(x) = [x[1] - 0.7, x[1] - 0.2, x[end] + 0.2, x[end] + 0.7]

# Composite Simpson's rule over the knot intervals, cumulative from x[1]. It
# is exact for piecewise cubics with breakpoints at the knots, so it gives an
# independent check of the cumulative integrals stored at construction.
function simpson_cumulative(f, x)
    I = zeros(length(x))
    for i in 2:length(x)
        a, b = x[i-1], x[i]
        fa, fm, fb = f([a, (a + b) / 2, b])
        I[i] = I[i-1] + (b - a) / 6 * (fa + 4fm + fb)
    end
    return I
end

# Jumps in value, first, and second derivative at each interior knot, computed
# from the segment polynomials y = a + b dx + c dx^2 + d dx^3. Row k holds the
# (k-1)th-derivative jumps; a C^k interpolant has zeros in rows 1:k+1.
function knot_jumps(s)
    x, a, b, c, d = s.x, s.a, s.b, s.c, s.d
    n = length(x)
    jumps = zeros(3, n - 2)
    for i in 2:n-1
        h = x[i] - x[i-1]
        jumps[1, i-1] = abs(a[i-1] + b[i-1] * h + c[i-1] * h^2 + d[i-1] * h^3 - a[i])
        jumps[2, i-1] = abs(b[i-1] + 2c[i-1] * h + 3d[i-1] * h^2 - b[i])
        jumps[3, i-1] = abs(2c[i-1] + 6d[i-1] * h - 2c[i])
    end
    return jumps
end

# Max-norm error of the interpolant built by `ctor(x, y)` for `f` on n uniform
# knots over [a, b].
function max_error(ctor, f, a, b, n; nq=1001)
    x = collect(range(a, b, n))
    xq = collect(range(a, b, nq))
    return maximum(abs.(evaluate_spline(ctor(x, f.(x)), xq) .- f.(xq)))
end

# Ratios of successive errors when the knot count doubles; a method of order p
# gives ratios approaching 2^p.
function convergence_ratios(ctor, f, a, b, ns)
    errs = [max_error(ctor, f, a, b, n) for n in ns]
    return errs[1:end-1] ./ errs[2:end]
end

# Shared checks for the evaluate / derivative / antiderivative interface of the
# one-dimensional interpolants, inside and outside the knot range.
function test_evaluation_interface(ctor, x, y)
    s = ctor(x, y)
    xi = interior_points(x)
    xo = outside_points(x)

    @testset "derivative matches finite differences" begin
        for xq in (xi, xo)
            fd = fd_derivative(q -> evaluate_spline(s, q), xq)
            @test evaluate_spline_derivative(s, xq) ≈ fd atol=1e-6
        end
    end

    @testset "antiderivative" begin
        # Zero at the first knot by default; `C` shifts every value.
        @test evaluate_spline_antiderivative(s, [x[1]]) == [0.0]
        @test evaluate_spline_antiderivative(s, xi; C=2.5) ≈ evaluate_spline_antiderivative(s, xi) .+ 2.5
        # Cumulative integrals at the knots agree with Simpson's rule, which is
        # exact for piecewise cubics.
        @test s.I ≈ simpson_cumulative(q -> evaluate_spline(s, q), x) atol=1e-12
        @test evaluate_spline_antiderivative(s, x) ≈ s.I atol=1e-12
        # Its derivative is the interpolant, inside and outside the knots.
        for xq in (xi, xo)
            fd = fd_derivative(q -> evaluate_spline_antiderivative(s, q), xq)
            @test evaluate_spline(s, xq) ≈ fd atol=1e-6
        end
    end

    @testset "extrapolate=false" begin
        s0 = ctor(x, y; extrapolate=false)
        for evalfn in (evaluate_spline, evaluate_spline_derivative, evaluate_spline_antiderivative)
            @test all(isnan, evalfn(s0, xo))
            @test !any(isnan, evalfn(s0, [x[1]; xi; x[end]]))  # the end knots are inside
            @test evalfn(s0, xi) == evalfn(s, xi)
        end
    end

    @testset "in-place evaluation" begin
        xq = [xi; xo]
        for (evalfn!, evalfn) in ((evaluate_spline!, evaluate_spline),
                                  (evaluate_spline_derivative!, evaluate_spline_derivative),
                                  (evaluate_spline_antiderivative!, evaluate_spline_antiderivative))
            out = fill(NaN, length(xq))
            @test evalfn!(out, s, xq) === out
            @test out == evalfn(s, xq)
            @test_throws DimensionMismatch evalfn!(zeros(length(xq) - 1), s, xq)
        end
        @test evaluate_spline(s, Float64[]) == Float64[]
    end
end
