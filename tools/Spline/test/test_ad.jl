using ForwardDiff

@testset "AD-safe interpolants" begin
    @testset "safe_spline matches CubicSpline (not-a-knot)" begin
        s = CubicSpline(REF_X, REF_Y)
        expected = evaluate_spline(s, REF_XQ)
        @test safe_spline(REF_X, REF_Y, REF_XQ) ≈ expected atol=1e-12
        @test [safe_spline(REF_X, REF_Y, q) for q in REF_XQ] ≈ expected atol=1e-12
        itp = safe_spline(REF_X, REF_Y)
        @test itp.(REF_XQ) ≈ expected atol=1e-12
        @test itp.(REF_X) ≈ REF_Y atol=1e-12
        @test_throws DimensionMismatch safe_spline([0.0, 1.0, 2.0], [0.0, 1.0])
        @test_throws ArgumentError safe_spline([0.0, 1.0], [0.0, 1.0])
    end

    @testset "safe_pchip matches PchipSpline" begin
        s = PchipSpline(REF_X, REF_Y)
        expected = evaluate_spline(s, REF_XQ)
        @test safe_pchip(REF_X, REF_Y, REF_XQ) ≈ expected atol=1e-12
        @test [safe_pchip(REF_X, REF_Y, q) for q in REF_XQ] ≈ expected atol=1e-12
        itp = safe_pchip(REF_X, REF_Y)
        @test itp.(REF_XQ) ≈ expected atol=1e-12
        @test itp.(REF_X) ≈ REF_Y atol=1e-12
        s2 = PchipSpline(PCHIP2_X, PCHIP2_Y)
        @test safe_pchip(PCHIP2_X, PCHIP2_Y, PCHIP2_XQ) ≈ evaluate_spline(s2, PCHIP2_XQ) atol=1e-12
        # Two points: secant slopes, so the interpolant is the straight line.
        @test safe_pchip([0.0, 1.0], [0.0, 2.0], [0.5, 2.0, -1.0]) == [1.0, 4.0, -2.0]
        @test_throws DimensionMismatch safe_pchip([0.0, 1.0, 2.0], [0.0, 1.0])
        @test_throws ArgumentError safe_pchip([0.0], [0.0])
    end

    @testset "derivative with respect to the query point ($name)" for (name, x) in GRIDS
        y = f_smooth.(x)
        xq = [outside_points(x); interior_points(x)]
        itp = safe_spline(x, y)
        @test [ForwardDiff.derivative(itp, q) for q in xq] ≈ evaluate_spline_derivative(CubicSpline(x, y), xq) atol=1e-12
        itp = safe_pchip(x, y)
        @test [ForwardDiff.derivative(itp, q) for q in xq] ≈ evaluate_spline_derivative(PchipSpline(x, y), xq) atol=1e-12
    end

    @testset "gradient with respect to the data ($name)" for (name, x) in GRIDS
        y = f_smooth.(x)
        n = length(x)
        for interp in (safe_spline, safe_pchip)
            # At a knot the interpolant equals y[k], so the gradient is e_k.
            for k in (1, 3, n)
                g = ForwardDiff.gradient(yy -> interp(x, yy, x[k]), y)
                @test g ≈ [i == k for i in 1:n] atol=1e-12
            end
            # Adding a constant to the data shifts the interpolant by that
            # constant, so the gradient sums to one at every query point.
            for q in [outside_points(x); interior_points(x)]
                @test sum(ForwardDiff.gradient(yy -> interp(x, yy, q), y)) ≈ 1 atol=1e-10
            end
        end
    end

    @testset "derivative with respect to the knots ($name)" for (name, x) in GRIDS
        y = f_smooth.(x)
        # Shifting all knots right by t is the same as querying at xq - t.
        for (interp, ctor) in ((safe_spline, CubicSpline), (safe_pchip, PchipSpline))
            dref = evaluate_spline_derivative(ctor(x, y), interior_points(x))
            for (k, q) in enumerate(interior_points(x))
                d = ForwardDiff.derivative(t -> interp(x .+ t, y, q), 0.0)
                @test d ≈ -dref[k] atol=1e-10
            end
        end
    end

    @testset "generic element types" begin
        xi = collect(0:4)              # integer knots
        xr = range(0.0, 4.0, length=5)  # range knots
        y = p3.(xi)
        for interp in (safe_spline, safe_pchip)
            @test interp(xi, y, 1.5) ≈ interp(Float64.(xi), y, 1.5)
            @test interp(xr, y, 1.5) ≈ interp(collect(xr), y, 1.5)
            @test interp(Float32.(xi), Float32.(y), 1.5f0) isa Float32
        end
        # Not-a-knot reproduces the cubic, whatever the knot container.
        @test safe_spline(xi, y, [0.5, 2.25, 5.0]) ≈ p3.([0.5, 2.25, 5.0]) atol=1e-12
    end
end
