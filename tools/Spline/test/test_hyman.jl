@testset "HymanSpline" begin
    @testset "input validation" begin
        @test_throws DimensionMismatch HymanSpline([0.0, 1.0, 2.0], [0.0, 1.0])
        @test_throws ArgumentError HymanSpline([0.0, 1.0], [0.0, 1.0])
        @test_throws ArgumentError HymanSpline([0.0, 2.0, 1.0], [0.0, 1.0, 2.0])
        @test_throws ArgumentError HymanSpline(DUP_X, DUP_Y)
        s = @test_logs (:warn, r"not recognized") HymanSpline(REF_X, REF_Y; bc_type="periodic")
        @test s.bc_type == "not-a-knot"
    end

    @testset "interpolates the data ($bc, $name)" for bc in BC_TYPES, (name, x) in GRIDS
        y = f_smooth.(x)
        @test evaluate_spline(HymanSpline(x, y; bc_type=bc), x) ≈ y atol=1e-12
    end

    @testset "C1 at the knots ($bc, $name)" for bc in BC_TYPES, (name, x) in GRIDS
        jumps = knot_jumps(HymanSpline(x, f_smooth.(x); bc_type=bc))
        @test all(jumps[1:2, :] .< 1e-12)
    end

    @testset "slopes satisfy the Fritsch-Carlson bounds ($name)" for (name, (x, y)) in MONOTONE_DATA
        s = HymanSpline(x, y)
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
        # The unconstrained cubic spline overshoots on this data, so the
        # check below is discriminating.
        @test minimum(evaluate_spline_derivative(CubicSpline(x, y), xq)) < -0.01
        @test minimum(evaluate_spline_derivative(HymanSpline(x, y), xq)) >= -1e-12
        @test maximum(evaluate_spline_derivative(HymanSpline(x, -y), xq)) <= 1e-12
        # Consequently every value lies between the neighbouring data values.
        v = evaluate_spline(HymanSpline(x, y), xq)
        @test all(minimum(y) - 1e-12 .<= v .<= maximum(y) + 1e-12)
    end

    @testset "zero slope at extrema and on flat segments" begin
        # Interior peak: the secants change sign at the third knot.
        s = HymanSpline([0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 3.0, 1.0, 0.0])
        @test s.m[3] == 0
        # Flat segments are reproduced exactly as constants.
        x = collect(0.0:5.0)
        y = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        s = HymanSpline(x, y)
        @test s.m == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        @test evaluate_spline(s, [0.3, 1.5, 1.9]) == [0.0, 0.0, 0.0]
        @test evaluate_spline(s, [3.1, 4.5, 4.9]) == [1.0, 1.0, 1.0]
        @test evaluate_spline_derivative(s, [0.3, 1.5, 4.5]) == [0.0, 0.0, 0.0]
    end

    @testset "coincides with CubicSpline when the filter is inactive ($bc)" for bc in BC_TYPES
        # Convex, increasing data on a fine grid: every cubic-spline slope is
        # already inside the Fritsch-Carlson region.
        x = collect(range(0.0, 2.0, 15))
        y = exp.(x)
        h = HymanSpline(x, y; bc_type=bc)
        c = CubicSpline(x, y; bc_type=bc)
        @test h.b ≈ c.b atol=1e-12
        @test h.c ≈ c.c[1:end-1] atol=1e-12
        @test h.d ≈ c.d atol=1e-12
        xq = [outside_points(x); interior_points(x)]
        @test evaluate_spline(h, xq) ≈ evaluate_spline(c, xq) atol=1e-12
        @test evaluate_spline_derivative(h, xq) ≈ evaluate_spline_derivative(c, xq) atol=1e-12
        @test evaluate_spline_antiderivative(h, xq) ≈ evaluate_spline_antiderivative(c, xq) atol=1e-12
    end

    @testset "coincides with CubicSpline on intervals whose slopes pass unchanged" begin
        # A coarse grid on a full period: the knots nearest the extrema are
        # filtered, the rest are not.
        x = collect(range(0.0, 2π, 11))
        y = sin.(x)
        h = HymanSpline(x, y)
        c = CubicSpline(x, y)
        unchanged = isapprox.(h.m, evaluate_spline_derivative(c, x); atol=1e-12)
        @test any(unchanged) && !all(unchanged)
        for i in 1:length(x)-1
            xq = x[i] .+ (x[i+1] - x[i]) .* [0.25, 0.5, 0.75]
            same = evaluate_spline(h, xq) ≈ evaluate_spline(c, xq)
            @test same == (unchanged[i] && unchanged[i+1])
        end
    end

    @testset "evaluation interface ($bc, $name)" for bc in BC_TYPES, (name, x) in GRIDS
        test_evaluation_interface((x, y; kw...) -> HymanSpline(x, y; bc_type=bc, kw...), x, f_smooth.(x))
    end

    @testset "fourth-order convergence on smooth data" begin
        ratios = convergence_ratios(HymanSpline, sin, 0.0, 2π, [21, 41, 81])
        @test all(ratios .> 12)
    end
end
