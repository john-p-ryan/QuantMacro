# Uses the 2-D data and helpers of test_bilinear.jl (BX, BY, BQ_X, BQ_Y, g_curved).

# Trilinear test function: reproduced exactly, including under linear extrapolation.
g_trilinear(x, y, z) = 1 + 2x - 3y + z + 0.5x * y - x * z + 2y * z + 0.25x * y * z
# Non-multilinear test function.
g_curved3(x, y, z) = x^2 + sin(y) + x * y^2 + exp(-z) * x
# Four-linear test function.
g_quadlinear(x, y, z, w) = 1 - x + 2y * w + x * z - 0.5y * z * w + 0.1x * y * z * w

# Reference multilinear interpolation by successive one-dimensional linear
# interpolation, contracting the leading dimension of the value array once per
# coordinate. Unit-cell offsets outside [0, 1] give linear extrapolation.
function reference_multilinear(knots, values, point)
    for (x, p) in zip(knots, point)
        i = clamp(searchsortedlast(x, p), 1, length(x) - 1)
        t = (p - x[i]) / (x[i+1] - x[i])
        values = (1 - t) .* selectdim(values, 1, i) .+ t .* selectdim(values, 1, i + 1)
    end
    return values[]
end

# Coordinates below, on, inside, and above the knot range of one dimension.
probe_points(x) = [x[1] - 0.7, x[1], x[1] + 0.3 * (x[2] - x[1]), x[2], (x[end-1] + x[end]) / 2, x[end], x[end] + 0.4]

# All combinations of the given coordinate vectors, as one query vector per dimension.
function product_points(coords...)
    pts = vec(collect(Iterators.product(coords...)))
    return ntuple(k -> getindex.(pts, k), length(coords))
end

inside_mask(knots, q) = reduce(.&, [k[1] .<= v .<= k[end] for (k, v) in zip(knots, q)])

const MX = [0.0, 1.0, 3.0]
const MY = [0.0, 2.0, 3.0, 7.0]
const MZ = [-1.0, 0.5, 2.0]
const MKNOTS = (MX, MY, MZ)
# Query points covering the interior, faces, edges, corners, and every outside region.
const MQ = product_points(probe_points(MX), probe_points(MY), probe_points(MZ))
const MQ_INSIDE = inside_mask(MKNOTS, MQ)

@testset "MultilinearSpline" begin
    v = [g_curved3(x, y, z) for x in MX, y in MY, z in MZ]

    @testset "input validation" begin
        @test_throws DimensionMismatch MultilinearSpline(MX, MY, MZ, permutedims(v, (2, 1, 3)))
        @test_throws DimensionMismatch MultilinearSpline(MX, MY, v)            # too few grid vectors
        @test_throws DimensionMismatch MultilinearSpline(MX, MY, MZ, v[:, :, 1])  # too many
        @test_throws DimensionMismatch MultilinearSpline((MX, MY, MZ), v[:, :, 1:2])
        @test_throws ArgumentError MultilinearSpline([0.0], MY, MZ, v[1:1, :, :])
        @test_throws ArgumentError MultilinearSpline(MX, MY, [0.0], v[:, :, 1:1])
        @test_throws ArgumentError MultilinearSpline(reverse(MX), MY, MZ, v)
        @test_throws ArgumentError MultilinearSpline(MX, MY, reverse(MZ), v)
        @test_throws ArgumentError MultilinearSpline(MX, MY, MZ, v; bc_type="cubic")
        @test_throws ArgumentError MultilinearSpline([0.0, 1.0, 1.0], MY, MZ, v)  # duplicate knots
        @test_throws ArgumentError MultilinearSpline((), fill(1.0))               # zero dimensions
    end

    @testset "tuple and vararg constructors agree" begin
        s1 = MultilinearSpline(MX, MY, MZ, v; bc_type="constant", extrapolate=false)
        s2 = MultilinearSpline((MX, MY, MZ), v; bc_type="constant", extrapolate=false)
        @test s1.knots === s2.knots && s1.values === s2.values
        @test s1.bc_type == s2.bc_type && s1.extrapolate == s2.extrapolate
    end

    @testset "reproduces trilinear functions" begin
        vt = [g_trilinear(x, y, z) for x in MX, y in MY, z in MZ]
        s = MultilinearSpline(MX, MY, MZ, vt)
        @test evaluate_spline(s, MQ...) ≈ g_trilinear.(MQ...) atol=1e-12
        gs = (probe_points(MX), probe_points(MY), probe_points(MZ))
        @test evaluate_spline_grid(s, gs...) ≈ [g_trilinear(x, y, z) for x in gs[1], y in gs[2], z in gs[3]] atol=1e-12
    end

    @testset "interpolates the nodes and cell centres" begin
        s = MultilinearSpline(MX, MY, MZ, v)
        nodes = product_points(MX, MY, MZ)
        @test evaluate_spline(s, nodes...) ≈ vec(v) atol=1e-12
        @test evaluate_spline_grid(s, MX, MY, MZ) ≈ v atol=1e-12
        for i in 1:length(MX)-1, j in 1:length(MY)-1, k in 1:length(MZ)-1
            centre = ((MX[i] + MX[i+1]) / 2, (MY[j] + MY[j+1]) / 2, (MZ[k] + MZ[k+1]) / 2)
            corners = sum(v[i:i+1, j:j+1, k:k+1]) / 8
            @test Spline.evaluate_point(s, centre...) ≈ corners atol=1e-12
        end
    end

    @testset "matches the reference implementation" begin
        s = MultilinearSpline(MX, MY, MZ, v)
        ref = [reference_multilinear(MKNOTS, v, p) for p in zip(MQ...)]
        @test evaluate_spline(s, MQ...) ≈ ref atol=1e-12
    end

    @testset "reduces to bilinear interpolation on knot planes" begin
        s = MultilinearSpline(MX, MY, MZ, v)
        yz = product_points(interior_points(MY), interior_points(MZ))
        for (i, x) in enumerate(MX)
            b = BilinearSpline(MY, MZ, v[i, :, :])
            @test evaluate_spline(s, fill(x, length(yz[1])), yz...) ≈ evaluate_spline(b, yz...) atol=1e-12
        end
        xy = product_points(interior_points(MX), interior_points(MY))
        for (k, z) in enumerate(MZ)
            b = BilinearSpline(MX, MY, v[:, :, k])
            @test evaluate_spline(s, xy..., fill(z, length(xy[1]))) ≈ evaluate_spline(b, xy...) atol=1e-12
        end
    end

    @testset "two dimensions agree with BilinearSpline ($bc, extrapolate=$ex)" for bc in ["linear", "constant"], ex in [true, false]
        z = [g_curved(x, y) for x in BX, y in BY]
        b = BilinearSpline(BX, BY, z; bc_type=bc, extrapolate=ex)
        m = MultilinearSpline(BX, BY, z; bc_type=bc, extrapolate=ex)
        @test isapprox(evaluate_spline(m, BQ_X, BQ_Y), evaluate_spline(b, BQ_X, BQ_Y); atol=1e-12, nans=true)
        gx = [-1.0, 0.0, 0.4, 1.0, 2.2, 3.0, 5.0]
        gy = [-2.0, 0.0, 1.3, 2.0, 2.9, 5.5, 7.0, 9.0]
        @test isapprox(evaluate_spline_grid(m, gx, gy), evaluate_spline_grid(b, gx, gy); atol=1e-12, nans=true)
    end

    @testset "one dimension agrees with LinearSpline ($name)" for (name, x) in GRIDS
        y = f_smooth.(x)
        xq = [outside_points(x); interior_points(x); x]
        @test evaluate_spline(MultilinearSpline(x, y), xq) ≈ evaluate_spline(LinearSpline(x, y), xq) atol=1e-12
        @test evaluate_spline_grid(MultilinearSpline(x, y), xq) ≈ evaluate_spline(LinearSpline(x, y), xq) atol=1e-12
        m0 = MultilinearSpline(x, y; extrapolate=false)
        @test isapprox(evaluate_spline(m0, xq), evaluate_spline(LinearSpline(x, y; extrapolate=false), xq); atol=1e-12, nans=true)
    end

    @testset "extrapolation" begin
        lin = MultilinearSpline(MX, MY, MZ, v; bc_type="linear")
        con = MultilinearSpline(MX, MY, MZ, v; bc_type="constant")
        off = MultilinearSpline(MX, MY, MZ, v; extrapolate=false)
        vl = evaluate_spline(lin, MQ...)
        vc = evaluate_spline(con, MQ...)
        vo = evaluate_spline(off, MQ...)
        # Inside the grid all three agree; outside, constant extrapolation
        # takes the value at the nearest point of the grid.
        @test vl[MQ_INSIDE] == vc[MQ_INSIDE] == vo[MQ_INSIDE]
        clamped = ntuple(k -> clamp.(MQ[k], MKNOTS[k][1], MKNOTS[k][end]), 3)
        @test vc ≈ evaluate_spline(lin, clamped...) atol=1e-12
        @test all(isnan, vo[.!MQ_INSIDE])
        @test !any(isnan, vl) && !any(isnan, vc)
        # Linear extrapolation continues the edge cell's multilinear polynomial.
        @test vl ≈ [reference_multilinear(MKNOTS, v, p) for p in zip(MQ...)] atol=1e-12
    end

    @testset "grid evaluation matches pointwise ($bc, extrapolate=$ex)" for bc in ["linear", "constant"], ex in [true, false]
        s = MultilinearSpline(MX, MY, MZ, v; bc_type=bc, extrapolate=ex)
        gs = (probe_points(MX), probe_points(MY), probe_points(MZ))
        G = evaluate_spline_grid(s, gs...)
        @test size(G) == map(length, gs)
        pointwise = [evaluate_spline(s, [x], [y], [z])[1] for x in gs[1], y in gs[2], z in gs[3]]
        @test isequal(G, pointwise)  # NaN-aware
    end

    @testset "four dimensions" begin
        W = [0.0, 0.5, 1.0]
        v4 = [g_quadlinear(x, y, z, w) for x in MX, y in MY, z in MZ, w in W]
        s = MultilinearSpline(MX, MY, MZ, W, v4)
        q = product_points(probe_points(MX), probe_points(MY), probe_points(MZ), probe_points(W))
        @test evaluate_spline(s, q...) ≈ g_quadlinear.(q...) atol=1e-12
        gs = ([-0.5, 0.2, 1.0, 3.5], [1.0, 2.0, 8.0], [-1.0, 0.0, 2.5], [-0.1, 0.5, 0.7])
        @test evaluate_spline_grid(s, gs...) ≈ [g_quadlinear(p...) for p in Iterators.product(gs...)] atol=1e-12
        con = MultilinearSpline(MX, MY, MZ, W, v4; bc_type="constant")
        clamped = ntuple(k -> clamp.(q[k], (MX, MY, MZ, W)[k][1], (MX, MY, MZ, W)[k][end]), 4)
        @test evaluate_spline(con, q...) ≈ g_quadlinear.(clamped...) atol=1e-12
    end

    @testset "in-place evaluation" begin
        s = MultilinearSpline(MX, MY, MZ, v)
        out = fill(NaN, length(MQ[1]))
        @test evaluate_spline!(out, s, MQ...) === out
        @test out == evaluate_spline(s, MQ...)
        @test_throws DimensionMismatch evaluate_spline!(zeros(2), s, MQ...)
        @test_throws DimensionMismatch evaluate_spline(s, MQ[1], MQ[2], MQ[3][1:end-1])
        @test_throws MethodError evaluate_spline(s, MQ[1], MQ[2])  # wrong number of coordinates
        @test evaluate_spline(s, Float64[], Float64[], Float64[]) == Float64[]
    end

    @testset "type stability" begin
        s = MultilinearSpline(MX, MY, MZ, v)
        @test @inferred(MultilinearSpline(MX, MY, MZ, v)) isa MultilinearSplineInterpolation{3}
        @test @inferred(evaluate_spline(s, MQ...)) isa Vector{Float64}
        @test @inferred(evaluate_spline_grid(s, MX, MY, MZ)) isa Array{Float64,3}
        @test @inferred(Spline.evaluate_point(s, 0.5, 1.0, 0.0)) isa Float64
        @test @allocated(Spline.evaluate_point(s, 0.5, 1.0, 0.0)) == 0
    end
end
