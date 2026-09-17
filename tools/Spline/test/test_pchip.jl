@testset "PchipSpline" begin
    @testset "input validation" begin
        @test_throws DimensionMismatch PchipSpline([0.0, 1.0, 2.0], [0.0, 1.0])
        @test_throws ArgumentError PchipSpline([0.0], [0.0])
        @test_throws ArgumentError PchipSpline([0.0, 2.0, 1.0], [0.0, 1.0, 2.0])
        @test_throws ArgumentError PchipSpline(DUP_X, DUP_Y)  # duplicate knots
    end

    @testset "two points" begin
        # Secant slopes, so the interpolant is the straight line through the data.
        s = PchipSpline([0.0, 1.0], [0.0, 2.0])
        @test s.m == [2.0, 2.0]
        @test evaluate_spline(s, [0.0, 0.5, 1.0, 2.0, -1.0]) == [0.0, 1.0, 2.0, 4.0, -2.0]
        @test evaluate_spline_derivative(s, [0.5, 2.0]) == [2.0, 2.0]
        @test evaluate_spline_antiderivative(s, [1.0, 2.0]) == [1.0, 4.0]
    end

    @testset "agrees with SciPy" begin
        s = PchipSpline(REF_X, REF_Y)
        inside = REF_XQ[1:3]
        @test s.m ≈ SCIPY_PCHIP.slopes atol=1e-12
        @test evaluate_spline(s, inside) ≈ SCIPY_PCHIP.value atol=1e-12
        @test evaluate_spline_derivative(s, inside) ≈ SCIPY_PCHIP.deriv atol=1e-12
        @test evaluate_spline_antiderivative(s, inside) ≈ SCIPY_PCHIP.antideriv atol=1e-12
        # Endpoint clamps and sign changes in the secants.
        s2 = PchipSpline(PCHIP2_X, PCHIP2_Y)
        @test s2.m ≈ SCIPY_PCHIP2.slopes atol=1e-12
        @test evaluate_spline(s2, PCHIP2_XQ) ≈ SCIPY_PCHIP2.value atol=1e-12
    end

    @testset "endpoint slopes" begin
        # Three-point formula when no clamp is triggered.
        x, y = REF_X, REF_Y
        h = diff(x)
        delta = diff(y) ./ h
        s = PchipSpline(x, y)
        @test s.m[1] ≈ ((2h[1] + h[2]) * delta[1] - h[1] * delta[2]) / (h[1] + h[2])
        @test s.m[end] ≈ ((2h[end] + h[end-1]) * delta[end] - h[end] * delta[end-1]) / (h[end] + h[end-1])
        # The slope is set to zero when the formula disagrees in sign with the
        # first secant, and capped at three times it when the secants change
        # sign beyond the first interval.
        @test PchipSpline([0.0, 1.0, 2.0], [0.0, 1.0, 5.0]).m[1] == 0
        @test PchipSpline([0.0, 1.0, 2.0], [0.0, 0.1, -1.0]).m[1] ≈ 0.3
        @test PchipSpline([0.0, 1.0, 2.0], [5.0, 1.0, 0.0]).m[3] == 0
        @test PchipSpline([0.0, 1.0, 2.0], [-1.0, 0.1, 0.0]).m[3] ≈ -0.3
    end

    @testset "interpolates the data ($name)" for (name, x) in GRIDS
        y = f_smooth.(x)
        @test evaluate_spline(PchipSpline(x, y), x) ≈ y atol=1e-12
    end

    @testset "C1 at the knots ($name)" for (name, x) in GRIDS
        jumps = knot_jumps(PchipSpline(x, f_smooth.(x)))
        @test all(jumps[1:2, :] .< 1e-12)
    end

    @testset "reproduces linear functions ($name)" for (name, x) in GRIDS
        s = PchipSpline(x, p1.(x))
        xq = [outside_points(x); interior_points(x); x]
        @test evaluate_spline(s, xq) ≈ p1.(xq) atol=1e-12
        @test evaluate_spline_derivative(s, xq) ≈ fill(-0.5, length(xq)) atol=1e-12
        @test evaluate_spline_antiderivative(s, xq) ≈ P1.(xq) .- P1(x[1]) atol=1e-12
    end

    @testset "slopes satisfy the Fritsch-Carlson bounds ($name)" for (name, (x, y)) in MONOTONE_DATA
        s = PchipSpline(x, y)
        delta = diff(y) ./ diff(x)
        for i in eachindex(delta)
            if delta[i] == 0
                @test s.m[i] == 0 && s.m[i+1] == 0
            else
                @test 0 <= s.m[i] / delta[i] <= 3
                @test 0 <= s.m[i+1] / delta[i] <= 3
            end
        end
    end

    @testset "monotone where the data is monotone ($name)" for (name, (x, y)) in MONOTONE_DATA
        xq = collect(range(x[1], x[end], 2001))
        @test minimum(evaluate_spline_derivative(PchipSpline(x, y), xq)) >= -1e-12
        @test maximum(evaluate_spline_derivative(PchipSpline(x, -y), xq)) <= 1e-12
        v = evaluate_spline(PchipSpline(x, y), xq)
        @test all(minimum(y) - 1e-12 .<= v .<= maximum(y) + 1e-12)
    end

    @testset "zero slope at extrema and on flat segments" begin
        s = PchipSpline([0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 3.0, 1.0, 0.0])
        @test s.m[3] == 0
        x = collect(0.0:5.0)
        y = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        s = PchipSpline(x, y)
        @test s.m == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        @test evaluate_spline(s, [0.3, 1.5, 1.9]) == [0.0, 0.0, 0.0]
        @test evaluate_spline(s, [3.1, 4.5, 4.9]) == [1.0, 1.0, 1.0]
        @test evaluate_spline_derivative(s, [0.3, 1.5, 4.5]) == [0.0, 0.0, 0.0]
    end

    @testset "linear extrapolation with the endpoint slopes ($name)" for (name, x) in GRIDS
        y = f_smooth.(x)
        s = PchipSpline(x, y)
        t = [0.2, 0.7]
        @test evaluate_spline(s, x[1] .- t) ≈ y[1] .- s.m[1] .* t
        @test evaluate_spline(s, x[end] .+ t) ≈ y[end] .+ s.m[end] .* t
        @test evaluate_spline_derivative(s, x[1] .- t) == [s.m[1], s.m[1]]
        @test evaluate_spline_derivative(s, x[end] .+ t) == [s.m[end], s.m[end]]
    end

    @testset "evaluation interface ($name)" for (name, x) in GRIDS
        test_evaluation_interface(PchipSpline, x, f_smooth.(x))
    end

    @testset "at least second-order convergence" begin
        ratios = convergence_ratios(PchipSpline, sin, 0.0, 2π, [21, 41, 81])
        @test all(ratios .> 3.5)
    end
end
