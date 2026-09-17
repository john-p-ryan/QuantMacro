# Uses the helpers of test_bilinear.jl and test_multilinear.jl (g_bilinear,
# probe_points, product_points, inside_mask).

# Smooth, non-separable test function.
g_smooth2(x, y) = exp(-0.3x) * sin(1.5y) + 0.2x * y^2
# Separable factors: positive and increasing, so their product is monotone too.
g_sep_x(x) = 1 + x^2 / 4
g_sep_y(y) = 2 + tanh(y - 1)

# Reference evaluation of the piecewise cubic Hermite through (x, v) with knot
# slopes m, from the segment coefficients the 1-D splines use, and with the
# linear extrapolation of PchipSpline outside the knots.
function hermite_eval(x, v, m, xq)
    n = length(x)
    xq < x[1] && return v[1] + m[1] * (xq - x[1])
    xq > x[n] && return v[n] + m[n] * (xq - x[n])
    h = diff(x)
    b, c, d = Spline._hermite_coeffs(h, diff(v) ./ h, m)
    i = clamp(searchsortedlast(x, xq), 1, n - 1)
    dx = xq - x[i]
    return v[i] + b[i] * dx + c[i] * dx^2 + d[i] * dx^3
end

# Non-uniform grids with cells of very different sizes.
const PX = [0.0, 0.4, 1.0, 1.7, 3.0, 3.2]
const PY = [-1.0, 0.0, 0.5, 2.0, 2.4]
# Query points covering the interior, edges, corners, and every outside region.
const PQ = product_points(probe_points(PX), probe_points(PY))
const PQ_INSIDE = inside_mask((PX, PY), PQ)

@testset "Pchip2DSpline" begin
    z = [g_smooth2(x, y) for x in PX, y in PY]

    @testset "input validation" begin
        @test_throws DimensionMismatch Pchip2DSpline(PX, PY, permutedims(z))
        @test_throws ArgumentError Pchip2DSpline([0.0], PY, z[1:1, :])
        @test_throws ArgumentError Pchip2DSpline(PX, [0.0], z[:, 1:1])
        @test_throws ArgumentError Pchip2DSpline(reverse(PX), PY, z)
        @test_throws ArgumentError Pchip2DSpline(PX, reverse(PY), z)
        @test_throws ArgumentError Pchip2DSpline(PX, PY, z; bc_type="cubic")
        @test_throws ArgumentError Pchip2DSpline([0.0, 1.0, 1.0], [0.0, 1.0], ones(3, 2))  # duplicate knots
        @test_throws ArgumentError Pchip2DSpline([0.0, 1.0], [0.0, 1.0, 1.0], ones(2, 3))
    end

    @testset "node data are the 1-D PCHIP slopes" begin
        s = Pchip2DSpline(PX, PY, z)
        @test s.z === z
        for j in eachindex(PY)
            @test s.zx[:, j] == PchipSpline(PX, z[:, j]).m
        end
        for i in eachindex(PX)
            @test s.zy[i, :] == PchipSpline(PY, z[i, :]).m
        end
        # Cross derivatives: average of the PCHIP slopes of zx along y and of zy along x.
        zxy = [PchipSpline(PY, s.zx[i, :]).m[j] for i in eachindex(PX), j in eachindex(PY)]
        zyx = [PchipSpline(PX, s.zy[:, j]).m[i] for i in eachindex(PX), j in eachindex(PY)]
        @test s.zxy ≈ (zxy .+ zyx) ./ 2 atol=1e-14
    end

    @testset "interpolates the nodes" begin
        s = Pchip2DSpline(PX, PY, z)
        nodes = product_points(PX, PY)
        @test evaluate_spline(s, nodes...) ≈ vec(z) atol=1e-12
        @test evaluate_spline_grid(s, PX, PY) ≈ z atol=1e-12
    end

    @testset "reproduces bilinear functions" begin
        zb = [g_bilinear(x, y) for x in PX, y in PY]
        s = Pchip2DSpline(PX, PY, zb)
        @test evaluate_spline(s, PQ...) ≈ g_bilinear.(PQ...) atol=1e-12
        gx, gy = probe_points(PX), probe_points(PY)
        @test evaluate_spline_grid(s, gx, gy) ≈ [g_bilinear(x, y) for x in gx, y in gy] atol=1e-12
    end

    @testset "a single cell is the bilinear interpolant" begin
        x2, y2, z2 = PX[2:3], PY[3:4], z[2:3, 3:4]
        q = product_points(probe_points(x2), probe_points(y2))
        @test evaluate_spline(Pchip2DSpline(x2, y2, z2), q...) ≈ evaluate_spline(BilinearSpline(x2, y2, z2), q...) atol=1e-12
    end

    @testset "reduces to PchipSpline along grid lines" begin
        s = Pchip2DSpline(PX, PY, z)
        # Inside and outside the knots, so the extrapolation agrees too.
        yq = [outside_points(PY); interior_points(PY)]
        for (i, x) in enumerate(PX)
            @test evaluate_spline(s, fill(x, length(yq)), yq) ≈ evaluate_spline(PchipSpline(PY, z[i, :]), yq) atol=1e-12
        end
        xq = [outside_points(PX); interior_points(PX)]
        for (j, y) in enumerate(PY)
            @test evaluate_spline(s, xq, fill(y, length(xq))) ≈ evaluate_spline(PchipSpline(PX, z[:, j]), xq) atol=1e-12
        end
    end

    @testset "separable data gives the product and sum of the 1-D PCHIPs" begin
        gx, gy = g_sep_x.(PX), g_sep_y.(PY)
        px = evaluate_spline(PchipSpline(PX, gx), PQ[1])
        py = evaluate_spline(PchipSpline(PY, gy), PQ[2])
        # Everywhere, including the extrapolation regions.
        @test evaluate_spline(Pchip2DSpline(PX, PY, gx * gy'), PQ...) ≈ px .* py atol=1e-12
        @test evaluate_spline(Pchip2DSpline(PX, PY, gx .+ gy'), PQ...) ≈ px .+ py atol=1e-12
    end

    @testset "symmetric in x and y" begin
        s = Pchip2DSpline(PX, PY, z)
        t = Pchip2DSpline(PY, PX, permutedims(z))
        @test evaluate_spline(s, PQ...) ≈ evaluate_spline(t, PQ[2], PQ[1]) atol=1e-12
        gx, gy = probe_points(PX), probe_points(PY)
        @test evaluate_spline_grid(s, gx, gy) ≈ permutedims(evaluate_spline_grid(t, gy, gx)) atol=1e-12
    end

    @testset "tensor-product structure" begin
        # Successive one-dimensional Hermite interpolation of the node data, in
        # either order, reproduces the interpolant everywhere: for fixed y it is
        # the cubic Hermite in x through the values f(x[i], y) and slopes
        # f_x(x[i], y), which are themselves cubic Hermites in y.
        s = Pchip2DSpline(PX, PY, z)
        ref_yx = [begin
            v = [hermite_eval(PY, s.z[i, :], s.zy[i, :], yq) for i in eachindex(PX)]
            m = [hermite_eval(PY, s.zx[i, :], s.zxy[i, :], yq) for i in eachindex(PX)]
            hermite_eval(PX, v, m, xq)
        end for (xq, yq) in zip(PQ...)]
        ref_xy = [begin
            v = [hermite_eval(PX, s.z[:, j], s.zx[:, j], xq) for j in eachindex(PY)]
            m = [hermite_eval(PX, s.zy[:, j], s.zxy[:, j], xq) for j in eachindex(PY)]
            hermite_eval(PY, v, m, yq)
        end for (xq, yq) in zip(PQ...)]
        @test evaluate_spline(s, PQ...) ≈ ref_yx atol=1e-12
        @test evaluate_spline(s, PQ...) ≈ ref_xy atol=1e-12
    end

    @testset "C1 across the knot lines" begin
        s = Pchip2DSpline(PX, PY, z)
        f(x, y) = evaluate_spline(s, [x], [y])[1]
        ε = 1e-6
        # Cross every knot line at interior and outside points of the other
        # coordinate; the grid boundary is where the cubic meets the linear
        # extension. The value is continuous, the one-sided difference
        # quotients agree, and the central one matches the slope of the node
        # data interpolated along the knot line.
        for (i, x) in enumerate(PX), y in [outside_points(PY); interior_points(PY)]
            fl, f0, fr = f(x - ε, y), f(x, y), f(x + ε, y)
            @test abs(fl - f0) < 1e-4 && abs(fr - f0) < 1e-4
            @test (f0 - fl) / ε ≈ (fr - f0) / ε atol=1e-4
            @test (fr - fl) / 2ε ≈ hermite_eval(PY, s.zx[i, :], s.zxy[i, :], y) atol=1e-4
        end
        for (j, y) in enumerate(PY), x in [outside_points(PX); interior_points(PX)]
            fl, f0, fr = f(x, y - ε), f(x, y), f(x, y + ε)
            @test abs(fl - f0) < 1e-4 && abs(fr - f0) < 1e-4
            @test (f0 - fl) / ε ≈ (fr - f0) / ε atol=1e-4
            @test (fr - fl) / 2ε ≈ hermite_eval(PX, s.zy[:, j], s.zxy[:, j], x) atol=1e-4
        end
    end

    @testset "monotone along grid lines" begin
        # Monotone in both directions but not separable, so only the grid
        # lines (where the interpolant is the 1-D PCHIP) are guaranteed.
        zm = (STEP_Y .+ RPN14_Y') .^ 2
        s = Pchip2DSpline(STEP_X, RPN14_X, zm)
        yq = collect(range(RPN14_X[1], RPN14_X[end], 1001))
        for (i, x) in enumerate(STEP_X)
            v = evaluate_spline(s, fill(x, length(yq)), yq)
            @test all(diff(v) .>= -1e-12)
            @test all(minimum(zm[i, :]) - 1e-12 .<= v .<= maximum(zm[i, :]) + 1e-12)
        end
        xq = collect(range(STEP_X[1], STEP_X[end], 1001))
        for (j, y) in enumerate(RPN14_X)
            v = evaluate_spline(s, xq, fill(y, length(xq)))
            @test all(diff(v) .>= -1e-12)
            @test all(minimum(zm[:, j]) - 1e-12 .<= v .<= maximum(zm[:, j]) + 1e-12)
        end
    end

    @testset "monotone everywhere for monotone separable data" begin
        # The product of two positive increasing PCHIPs is increasing in both arguments.
        zm = (1 .+ STEP_Y) * (1 .+ RPN14_Y)'
        s = Pchip2DSpline(STEP_X, RPN14_X, zm)
        G = evaluate_spline_grid(s, collect(range(STEP_X[1], STEP_X[end], 401)), collect(range(RPN14_X[1], RPN14_X[end], 401)))
        @test all(diff(G, dims=1) .>= -1e-12)
        @test all(diff(G, dims=2) .>= -1e-12)
        @test all(minimum(zm) - 1e-12 .<= G .<= maximum(zm) + 1e-12)
    end

    @testset "extrapolation" begin
        lin = Pchip2DSpline(PX, PY, z; bc_type="linear")
        con = Pchip2DSpline(PX, PY, z; bc_type="constant")
        off = Pchip2DSpline(PX, PY, z; extrapolate=false)
        vl = evaluate_spline(lin, PQ...)
        vc = evaluate_spline(con, PQ...)
        vo = evaluate_spline(off, PQ...)
        # Inside the grid all three agree; outside, constant extrapolation
        # takes the value at the nearest point of the grid.
        @test vl[PQ_INSIDE] == vc[PQ_INSIDE] == vo[PQ_INSIDE]
        cx = clamp.(PQ[1], PX[1], PX[end])
        cy = clamp.(PQ[2], PY[1], PY[end])
        @test vc ≈ evaluate_spline(lin, cx, cy) atol=1e-12
        @test all(isnan, vo[.!PQ_INSIDE])
        @test !any(isnan, vl) && !any(isnan, vc)

        # Past a side of the grid the interpolant is linear in the outside
        # coordinate, with the slope of the boundary node data interpolated
        # along the boundary.
        nx, ny = length(PX), length(PY)
        for y in interior_points(PY), (i, x0, dx) in ((nx, PX[end], 0.6), (1, PX[1], -0.6))
            f0, f1, f2 = evaluate_spline(lin, x0 .+ [0.0, dx, 2dx], fill(y, 3))
            @test f2 - f1 ≈ f1 - f0 atol=1e-12
            @test (f1 - f0) / dx ≈ hermite_eval(PY, lin.zx[i, :], lin.zxy[i, :], y) atol=1e-12
        end
        for x in interior_points(PX), (j, y0, dy) in ((ny, PY[end], 0.6), (1, PY[1], -0.6))
            f0, f1, f2 = evaluate_spline(lin, fill(x, 3), y0 .+ [0.0, dy, 2dy])
            @test f2 - f1 ≈ f1 - f0 atol=1e-12
            @test (f1 - f0) / dy ≈ hermite_eval(PX, lin.zy[:, j], lin.zxy[:, j], x) atol=1e-12
        end
        # Past a corner it is bilinear in the two offsets, with the corner node's derivatives.
        for (i, dx) in ((1, -0.8), (nx, 0.8)), (j, dy) in ((1, -0.5), (ny, 0.5))
            v = evaluate_spline(lin, [PX[i] + dx], [PY[j] + dy])[1]
            @test v ≈ z[i, j] + lin.zx[i, j] * dx + lin.zy[i, j] * dy + lin.zxy[i, j] * dx * dy atol=1e-12
        end
    end

    @testset "grid evaluation matches pointwise ($bc, extrapolate=$ex)" for bc in ["linear", "constant"], ex in [true, false]
        s = Pchip2DSpline(PX, PY, z; bc_type=bc, extrapolate=ex)
        gx, gy = probe_points(PX), probe_points(PY)
        G = evaluate_spline_grid(s, gx, gy)
        @test size(G) == (length(gx), length(gy))
        pointwise = [evaluate_spline(s, [x], [y])[1] for x in gx, y in gy]
        @test isequal(G, pointwise)  # NaN-aware
    end

    @testset "in-place evaluation" begin
        s = Pchip2DSpline(PX, PY, z)
        out = fill(NaN, length(PQ[1]))
        @test evaluate_spline!(out, s, PQ...) === out
        @test out == evaluate_spline(s, PQ...)
        @test_throws DimensionMismatch evaluate_spline!(zeros(2), s, PQ...)
        @test_throws DimensionMismatch evaluate_spline(s, PQ[1], PQ[2][1:end-1])
        @test evaluate_spline(s, Float64[], Float64[]) == Float64[]
    end

    @testset "at least second-order convergence" begin
        f(x, y) = sin(x) * exp(-0.2y) + 0.1x * y
        xf = collect(range(0.0, 2π, 301))
        yf = collect(range(0.0, 3.0, 201))
        exact = [f(x, y) for x in xf, y in yf]
        errs = map([11, 21, 41, 81]) do n
            x = collect(range(0.0, 2π, n))
            y = collect(range(0.0, 3.0, n))
            s = Pchip2DSpline(x, y, [f(a, b) for a in x, b in y])
            maximum(abs.(evaluate_spline_grid(s, xf, yf) .- exact))
        end
        # Doubling the knots in both directions divides a second-order error by 4.
        @test all(errs[1:end-1] ./ errs[2:end] .> 3.0)
    end

    @testset "type stability" begin
        s = Pchip2DSpline(PX, PY, z)
        @test @inferred(Pchip2DSpline(PX, PY, z)) isa Pchip2DSplineInterpolation
        @test @inferred(evaluate_spline(s, PQ...)) isa Vector{Float64}
        @test @inferred(evaluate_spline_grid(s, PX, PY)) isa Matrix{Float64}
        @test @inferred(Spline.evaluate_point(s, 0.5, 1.0)) isa Float64
        @test @allocated(Spline.evaluate_point(s, 0.5, 1.0)) == 0
    end
end
