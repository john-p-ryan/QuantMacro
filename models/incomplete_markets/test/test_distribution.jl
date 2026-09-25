@testset "wealth distribution" begin
    for (name, m) in METHODS
        @testset "$name" begin
            prim, res = baseline_household(m)
            n = length(dist_grid(prim))

            # Two stationary distributions from opposite corners of the state space.
            μ_poor = stationary_from(prim, res, 1, 1)
            μ_rich = stationary_from(prim, res, n, prim.nz)
            P = forward_operator(prim, res)

            @testset "transition matrix" begin
                @test size(P) == (n * prim.nz, n * prim.nz)
                @test all(nonzeros(P) .>= 0)
                k_next = clamp.(vec(dist_policy(prim, res)), prim.k_min, prim.k_max)
                for zp in 1:prim.nz
                    to_zp = P[z_block(prim, zp), :]
                    prob = repeat(prim.M[:, zp], inner=n)  # P(z' | z) for each source state (k, z)
                    # z' is drawn from M, and k' goes to at most two neighboring grid points in a lottery whose mean is
                    # the policy, so aggregate capital is preserved.
                    @test vec(sum(to_zp; dims=1)) ≈ prob
                    @test vec(dist_grid(prim)' * to_zp) ≈ prob .* k_next
                    @test all(count(!iszero, to_zp; dims=1) .<= max_support(prim))
                end
            end

            @testset "stationary distribution" begin
                @test all(μ_poor .>= 0) && sum(μ_poor) ≈ 1
                @test maxdiff(P * vec(μ_poor), vec(μ_poor)) < 1e-10
                @test vec(sum(μ_poor; dims=1)) ≈ stationary_z(prim)
                # Unique: the chain forgets its starting point.
                @test maxdiff(μ_poor, μ_rich) < 1e-9
                # The grid is wide enough that no mass piles up at k_max.
                @test sum(μ_poor[end, :]) < 1e-8
            end
        end
    end

    @testset "lottery weights on a small grid" begin
        for m in (EGM, InterpVFI)
            prim = m.Primitives(n_hist=5)
            res = m.initialize(prim)
            h = prim.k_hist
            # Savings below the grid, between two points, on a point, above the grid, and at the top.
            res.k_pol_hist = repeat([h[1] / 2, 0.75h[2] + 0.25h[3], h[4], h[5] + 1, h[5]], 1, prim.nz)
            expected = [[1, 0, 0, 0, 0], [0, 0.75, 0.25, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1]]
            P = m.build_T_star(prim, res)
            for z in 1:prim.nz, zp in 1:prim.nz, i in 1:5
                @test Vector(P[z_block(prim, zp), (z - 1) * 5 + i]) ≈ prim.M[z, zp] .* expected[i]
            end
        end
    end
end
