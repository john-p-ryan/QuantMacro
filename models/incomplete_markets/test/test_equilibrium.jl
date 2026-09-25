@testset "stationary equilibrium" begin
    ge = Dict(m => quietly(() -> m.solve_model()) for (_, m) in METHODS)

    for (name, m) in METHODS
        @testset "$name" begin
            prim, res = ge[m]

            @testset "markets clear" begin
                # After calculate_aggregates!, res.K is household capital supply and r, w are the firm's prices.
                @test res.K ≈ capital_demand(prim, res.r) rtol=market_tol(prim)
                @test res.w ≈ wage(prim, res.r)
                @test res.Y ≈ res.K^prim.α * prim.L^(1 - prim.α)
                # Goods market: aggregating budgets over a stationary distribution gives C + δK = rK + wL = Y, up to an
                # error second order in the capital-market gap.
                @test res.C + prim.δ * res.K ≈ res.Y rtol=1e-5
            end

            @testset "distribution" begin
                @test all(res.μ .>= 0) && sum(res.μ) ≈ 1
                @test vec(sum(res.μ; dims=1)) ≈ stationary_z(prim)
                @test sum(res.μ[end, :]) < 1e-4
            end

            @testset "precautionary saving (Aiyagari 1994)" begin
                # Uninsurable income risk raises saving, so r is below the complete-markets rate ρ + δ.
                below_complete_markets = res.r < complete_markets_r(prim)
                if m === GridSearch
                    # Known bias: on the default 500-point grid, the employed household's small saving increments near
                    # its target wealth round to k' = k (at k ≈ 7.7 near the true equilibrium r). Wealth accumulation
                    # stops too early, supply is too low, and r is pushed above ρ + δ. Refining the grid removes the
                    # bias only slowly (r ≈ 0.03496, as with EGM, needs nk ≈ 5000).
                    @test_broken below_complete_markets
                else
                    @test below_complete_markets
                end
                # Consumption smoothing: consumption is less dispersed than wealth.
                @test res.K_cv > res.C_cv > 0
            end
        end
    end

    @testset "methods agree" begin
        _, egm = ge[EGM]
        _, vfi = ge[InterpVFI]
        _, gs = ge[GridSearch]
        @test vfi.r ≈ egm.r atol=1e-5
        @test vfi.K ≈ egm.K rtol=2e-3
        @test vfi.K_cv ≈ egm.K_cv rtol=1e-2
        @test vfi.C_cv ≈ egm.C_cv rtol=1e-2
        @test gs.r ≈ egm.r atol=5e-4
        # The grid-search bias above truncates the upper tail of wealth, so dispersion is far too low.
        @test_broken gs.K_cv ≈ egm.K_cv rtol=0.1
    end

    @testset "risk aversion raises saving" begin
        # A higher γ strengthens the precautionary motive (prudence γ + 1), which raises K and lowers r.
        prim1, res1 = ge[EGM]
        _, res2 = quietly(() -> EGM.solve_model(γ=2.0))
        @test res2.r < res1.r < complete_markets_r(prim1)
        @test res2.K > res1.K
    end
end
