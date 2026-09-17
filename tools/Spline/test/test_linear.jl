@testset "LinearSpline" begin
    @testset "input validation" begin
        @test_throws DimensionMismatch LinearSpline([0.0, 1.0, 2.0], [0.0, 1.0])
        @test_throws ArgumentError LinearSpline([0.0], [0.0])
        @test_throws ArgumentError LinearSpline([0.0, 2.0, 1.0], [0.0, 1.0, 2.0])
        @test_throws ArgumentError LinearSpline(DUP_X, DUP_Y)  # duplicate knots
    end

    @testset "two points" begin
        s = LinearSpline([0.0, 1.0], [0.0, 2.0])
        @test evaluate_spline(s, [0.0, 0.5, 1.0, 2.0, -1.0]) == [0.0, 1.0, 2.0, 4.0, -2.0]
        @test evaluate_spline_derivative(s, [0.5, 2.0]) == [2.0, 2.0]
        @test evaluate_spline_antiderivative(s, [1.0, 2.0]) == [1.0, 4.0]
    end

    @testset "interpolates the data ($name)" for (name, x) in GRIDS
        y = f_smooth.(x)
        s = LinearSpline(x, y)
        @test evaluate_spline(s, x) ≈ y atol=1e-12
        # Midpoints are averages of the neighbouring data.
        mid = (x[1:end-1] .+ x[2:end]) ./ 2
        @test evaluate_spline(s, mid) ≈ (y[1:end-1] .+ y[2:end]) ./ 2 atol=1e-12
        # Values stay between the neighbouring data.
        xi = interior_points(x)
        v = evaluate_spline(s, xi)
        for (k, q) in enumerate(xi)
            i = searchsortedlast(x, q)
            @test min(y[i], y[i+1]) - 1e-12 <= v[k] <= max(y[i], y[i+1]) + 1e-12
        end
    end

    @testset "reproduces linear functions ($name)" for (name, x) in GRIDS
        s = LinearSpline(x, p1.(x))
        xq = [outside_points(x); interior_points(x); x]
        @test evaluate_spline(s, xq) ≈ p1.(xq) atol=1e-12
        @test evaluate_spline_derivative(s, xq) ≈ fill(-0.5, length(xq)) atol=1e-12
        @test evaluate_spline_antiderivative(s, xq) ≈ P1.(xq) .- P1(x[1]) atol=1e-12
    end

    @testset "piecewise-constant derivative ($name)" for (name, x) in GRIDS
        y = f_smooth.(x)
        s = LinearSpline(x, y)
        secant = diff(y) ./ diff(x)
        xi = interior_points(x)
        @test evaluate_spline_derivative(s, xi) == repeat(secant, inner=3)
        # Extrapolation continues the end segments.
        @test evaluate_spline_derivative(s, outside_points(x)) == [secant[1], secant[1], secant[end], secant[end]]
    end

    @testset "antiderivative is the trapezoid rule ($name)" for (name, x) in GRIDS
        y = f_smooth.(x)
        s = LinearSpline(x, y)
        trapz = cumsum([0.0; (y[1:end-1] .+ y[2:end]) ./ 2 .* diff(x)])
        @test evaluate_spline_antiderivative(s, x) ≈ trapz atol=1e-12
    end

    @testset "evaluation interface ($name)" for (name, x) in GRIDS
        test_evaluation_interface(LinearSpline, x, f_smooth.(x))
    end

    @testset "second-order convergence" begin
        ratios = convergence_ratios(LinearSpline, sin, 0.0, 2π, [21, 41, 81])
        @test all(3.5 .< ratios .< 4.5)
    end
end
