@testset "CubicSpline" begin
    @testset "input validation" begin
        @test_throws DimensionMismatch CubicSpline([0.0, 1.0, 2.0], [0.0, 1.0])
        @test_throws ArgumentError CubicSpline([0.0, 1.0], [0.0, 1.0])            # too few points
        @test_throws ArgumentError CubicSpline([0.0, 2.0, 1.0], [0.0, 1.0, 2.0])  # unsorted
        @test_throws ArgumentError CubicSpline(DUP_X, DUP_Y)                      # duplicate knots
        # An unrecognised boundary condition warns and falls back to not-a-knot.
        s = @test_logs (:warn, r"not recognized") CubicSpline(REF_X, REF_Y; bc_type="periodic")
        @test s.bc_type == "not-a-knot"
        @test s.b == CubicSpline(REF_X, REF_Y; bc_type="not-a-knot").b
        @test CubicSpline(REF_X, REF_Y; bc_type="Natural").bc_type == "natural"
    end

    @testset "agrees with SciPy ($bc)" for (bc, ref) in SCIPY_CUBIC
        s = CubicSpline(REF_X, REF_Y; bc_type=bc)
        @test evaluate_spline(s, REF_XQ) ≈ ref.value atol=1e-12
        @test evaluate_spline_derivative(s, REF_XQ) ≈ ref.deriv atol=1e-12
        @test evaluate_spline_antiderivative(s, REF_XQ) ≈ ref.antideriv atol=1e-12
    end

    @testset "interpolates the data ($bc, $name)" for bc in BC_TYPES, (name, x) in GRIDS
        y = f_smooth.(x)
        @test evaluate_spline(CubicSpline(x, y; bc_type=bc), x) ≈ y atol=1e-12
    end

    @testset "C2 at the knots ($bc, $name)" for bc in BC_TYPES, (name, x) in GRIDS
        s = CubicSpline(x, f_smooth.(x); bc_type=bc)
        @test all(knot_jumps(s) .< 1e-12)
    end

    @testset "boundary conditions ($name)" for (name, x) in GRIDS
        y = f_smooth.(x)
        n = length(x)
        h = diff(x)
        # Natural: zero second derivative at both ends.
        s = CubicSpline(x, y; bc_type="natural")
        @test s.c[1] == 0
        @test 2s.c[n-1] + 6s.d[n-1] * h[n-1] ≈ 0 atol=1e-12
        # Not-a-knot: the third derivative is continuous across the second and
        # penultimate knots, so the first two and last two cubics coincide.
        s = CubicSpline(x, y; bc_type="not-a-knot")
        @test s.d[1] ≈ s.d[2] atol=1e-12
        @test s.d[n-2] ≈ s.d[n-1] atol=1e-12
        # "clamped" is documented as not yet implemented and behaves as natural.
        @test CubicSpline(x, y; bc_type="clamped").b == CubicSpline(x, y; bc_type="natural").b
    end

    @testset "not-a-knot reproduces cubics ($name)" for (name, x) in GRIDS
        s = CubicSpline(x, p3.(x))
        xq = [outside_points(x); interior_points(x); x]  # extrapolation is the same cubic
        @test evaluate_spline(s, xq) ≈ p3.(xq) atol=1e-12
        @test evaluate_spline_derivative(s, xq) ≈ dp3.(xq) atol=1e-12
        @test evaluate_spline_antiderivative(s, xq) ≈ P3.(xq) .- P3(x[1]) atol=1e-12
    end

    @testset "natural reproduces linear functions ($name)" for (name, x) in GRIDS
        s = CubicSpline(x, p1.(x); bc_type="natural")
        xq = [outside_points(x); interior_points(x)]
        @test evaluate_spline(s, xq) ≈ p1.(xq) atol=1e-12
        @test evaluate_spline_derivative(s, xq) ≈ fill(-0.5, length(xq)) atol=1e-12
        @test evaluate_spline_antiderivative(s, xq) ≈ P1.(xq) .- P1(x[1]) atol=1e-12
    end

    @testset "evaluation interface ($bc, $name)" for bc in BC_TYPES, (name, x) in GRIDS
        test_evaluation_interface((x, y; kw...) -> CubicSpline(x, y; bc_type=bc, kw...), x, f_smooth.(x))
    end

    @testset "fourth-order convergence ($bc)" for bc in ["not-a-knot", "natural"]
        # sin has zero second derivative at 0 and 2π, so the natural spline
        # is also fourth order here.
        ratios = convergence_ratios((x, y) -> CubicSpline(x, y; bc_type=bc), sin, 0.0, 2π, [21, 41, 81])
        @test all(ratios .> 12)
    end

    @testset "factorization cache" begin
        x = collect(range(0.0, 1.0, 9))
        xq = [-0.3, 0.13, 0.5, 0.77, 1.4]
        CubicSpline(x, exp.(x))
        # Refit on the same grid reuses the cached factorization.
        @test evaluate_spline(CubicSpline(x, p3.(x)), xq) ≈ p3.(xq) atol=1e-12
        # A different grid of the same length must not hit a stale entry.
        x2 = x .^ 2
        @test evaluate_spline(CubicSpline(x2, p3.(x2)), xq) ≈ p3.(xq) atol=1e-12
        # The cache stores a copy of the knots, so mutating the caller's grid
        # after fitting cannot corrupt later fits.
        x .*= 2
        @test evaluate_spline(CubicSpline(x, p3.(x)), xq) ≈ p3.(xq) atol=1e-12
        # More distinct grids than the cache holds: eviction is harmless.
        for k in 1:70
            xk = collect(range(0.0, 1.0 + k / 100, 7))
            @test evaluate_spline(CubicSpline(xk, p3.(xk)), xq) ≈ p3.(xq) atol=1e-12
        end
    end
end
