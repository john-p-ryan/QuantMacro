@testset "primitives" begin
    for (name, m) in METHODS
        @testset "$name" begin
            prim = m.Primitives()

            @testset "grid" begin
                (; k_grid, k_min, k_max, nk) = prim
                @test length(k_grid) == nk
                @test k_grid[1] ≈ k_min && k_grid[end] ≈ k_max
                @test issorted(k_grid) && allunique(k_grid)
            end

            @testset "labor market" begin
                @test all(prim.M .>= 0) && vec(sum(prim.M; dims=2)) ≈ ones(prim.nz)
                # unemp is the long-run unemployment rate, and L the matching effective labor supply.
                @test stationary_z(prim) ≈ [1 - prim.unemp, prim.unemp]
                @test prim.L ≈ prim.ē * (stationary_z(prim) ⋅ prim.z_grid)
            end

            @testset "keyword overrides rebuild the grid" begin
                small = m.Primitives(nk=40, k_max=50.0)
                @test length(small.k_grid) == 40 && small.k_grid[end] ≈ 50.0
                @test_throws AssertionError m.Primitives(k_grid=collect(range(0.0, 1.0, 10)))
            end
        end
    end

    @testset "policy and histogram grids" begin
        for m in (EGM, InterpVFI)
            prim = m.Primitives()
            # density > 1 clusters policy grid points near the borrowing constraint, where the policies bend.
            @test all(diff(diff(prim.k_grid)) .> 0)
            @test prim.k_hist ≈ range(prim.k_min, prim.k_max, prim.n_hist)
            small = m.Primitives(n_hist=50, k_max=50.0)
            @test length(small.k_hist) == 50 && small.k_hist[end] ≈ 50.0
        end
    end
end
