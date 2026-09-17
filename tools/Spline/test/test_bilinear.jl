# Bilinear test function: reproduced exactly, including under linear extrapolation.
g_bilinear(x, y) = 1 + 2x - 3y + 0.5x * y
# Non-bilinear test function.
g_curved(x, y) = x^2 + sin(y) + x * y^2

# The bilinear polynomial of cell (i, j) evaluated at (x, y), from the corner values.
function g_bilinear_cell(X, Y, z, i, j, x, y)
    tx = (x - X[i]) / (X[i+1] - X[i])
    ty = (y - Y[j]) / (Y[j+1] - Y[j])
    return (1 - tx) * (1 - ty) * z[i, j] + tx * (1 - ty) * z[i+1, j] +
           (1 - tx) * ty * z[i, j+1] + tx * ty * z[i+1, j+1]
end

const BX = [0.0, 1.0, 3.0]
const BY = [0.0, 2.0, 3.0, 7.0]
# Query points covering the interior, edges, corners, and every outside region.
const BQ_X = [0.5, 2.0, 1.0, 0.0, 3.0, -1.0, 4.0, 1.5, 1.5, -1.0, 4.0, -0.5, 3.5]
const BQ_Y = [1.0, 2.5, 3.0, 0.0, 7.0, 2.5, 2.5, -1.0, 8.0, -1.0, 8.0, 7.5, -2.0]
const BQ_INSIDE = (BX[1] .<= BQ_X .<= BX[end]) .& (BY[1] .<= BQ_Y .<= BY[end])

@testset "BilinearSpline" begin
    z = [g_curved(x, y) for x in BX, y in BY]

    @testset "input validation" begin
        @test_throws DimensionMismatch BilinearSpline(BX, BY, permutedims(z))
        @test_throws ArgumentError BilinearSpline([0.0], BY, z[1:1, :])
        @test_throws ArgumentError BilinearSpline(BX, [0.0], z[:, 1:1])
        @test_throws ArgumentError BilinearSpline(reverse(BX), BY, z)
        @test_throws ArgumentError BilinearSpline(BX, reverse(BY), z)
        @test_throws ArgumentError BilinearSpline(BX, BY, z; bc_type="cubic")
        @test_throws ArgumentError BilinearSpline([0.0, 1.0, 1.0], [0.0, 1.0], ones(3, 2))  # duplicate knots
        @test_throws ArgumentError BilinearSpline([0.0, 1.0], [0.0, 1.0, 1.0], ones(2, 3))
    end

    @testset "reproduces bilinear functions" begin
        zb = [g_bilinear(x, y) for x in BX, y in BY]
        s = BilinearSpline(BX, BY, zb)
        @test evaluate_spline(s, BQ_X, BQ_Y) ≈ g_bilinear.(BQ_X, BQ_Y) atol=1e-12
        gx = [-1.0, 0.0, 0.4, 1.0, 2.2, 3.0, 5.0]
        gy = [-2.0, 0.0, 1.3, 2.0, 2.9, 5.5, 7.0, 9.0]
        @test evaluate_spline_grid(s, gx, gy) ≈ [g_bilinear(x, y) for x in gx, y in gy] atol=1e-12
    end

    @testset "interpolates the nodes and cell centres" begin
        s = BilinearSpline(BX, BY, z)
        nodes = [(x, y) for x in BX, y in BY]
        @test evaluate_spline(s, first.(vec(nodes)), last.(vec(nodes))) ≈ vec(z) atol=1e-12
        @test evaluate_spline_grid(s, BX, BY) ≈ z atol=1e-12
        for i in 1:length(BX)-1, j in 1:length(BY)-1
            cx, cy = (BX[i] + BX[i+1]) / 2, (BY[j] + BY[j+1]) / 2
            corners = (z[i, j] + z[i+1, j] + z[i, j+1] + z[i+1, j+1]) / 4
            @test evaluate_spline(s, [cx], [cy]) ≈ [corners] atol=1e-12
        end
    end

    @testset "reduces to linear interpolation along grid lines" begin
        s = BilinearSpline(BX, BY, z)
        yq = interior_points(BY)
        for (i, x) in enumerate(BX)
            @test evaluate_spline(s, fill(x, length(yq)), yq) ≈ evaluate_spline(LinearSpline(BY, z[i, :]), yq) atol=1e-12
        end
        xq = interior_points(BX)
        for (j, y) in enumerate(BY)
            @test evaluate_spline(s, xq, fill(y, length(xq))) ≈ evaluate_spline(LinearSpline(BX, z[:, j]), xq) atol=1e-12
        end
    end

    @testset "extrapolation" begin
        lin = BilinearSpline(BX, BY, z; bc_type="linear")
        con = BilinearSpline(BX, BY, z; bc_type="constant")
        off = BilinearSpline(BX, BY, z; extrapolate=false)
        vl = evaluate_spline(lin, BQ_X, BQ_Y)
        vc = evaluate_spline(con, BQ_X, BQ_Y)
        vo = evaluate_spline(off, BQ_X, BQ_Y)
        # Inside the grid all three agree; outside, constant extrapolation
        # takes the value at the nearest point of the grid.
        @test vl[BQ_INSIDE] == vc[BQ_INSIDE] == vo[BQ_INSIDE]
        cx = clamp.(BQ_X, BX[1], BX[end])
        cy = clamp.(BQ_Y, BY[1], BY[end])
        @test vc ≈ evaluate_spline(lin, cx, cy) atol=1e-12
        @test all(isnan, vo[.!BQ_INSIDE])
        @test !any(isnan, vl) && !any(isnan, vc)
        # Linear extrapolation continues the edge cell's bilinear polynomial.
        @test evaluate_spline(lin, [-1.0], [8.0]) ≈ [g_bilinear_cell(BX, BY, z, 1, 3, -1.0, 8.0)] atol=1e-12
        @test evaluate_spline(lin, [4.0], [-1.0]) ≈ [g_bilinear_cell(BX, BY, z, 2, 1, 4.0, -1.0)] atol=1e-12
    end

    @testset "grid evaluation matches pointwise ($bc, extrapolate=$ex)" for bc in ["linear", "constant"], ex in [true, false]
        s = BilinearSpline(BX, BY, z; bc_type=bc, extrapolate=ex)
        gx = [-1.0, 0.0, 0.4, 1.0, 2.2, 3.0, 5.0]
        gy = [-2.0, 0.0, 1.3, 2.0, 2.9, 5.5, 7.0, 9.0]
        G = evaluate_spline_grid(s, gx, gy)
        @test size(G) == (length(gx), length(gy))
        pointwise = [evaluate_spline(s, [x], [y])[1] for x in gx, y in gy]
        @test isequal(G, pointwise)  # NaN-aware
    end

    @testset "in-place evaluation" begin
        s = BilinearSpline(BX, BY, z)
        out = fill(NaN, length(BQ_X))
        @test evaluate_spline!(out, s, BQ_X, BQ_Y) === out
        @test out == evaluate_spline(s, BQ_X, BQ_Y)
        @test_throws DimensionMismatch evaluate_spline!(zeros(2), s, BQ_X, BQ_Y)
        @test_throws DimensionMismatch evaluate_spline(s, BQ_X, BQ_Y[1:end-1])
    end
end
