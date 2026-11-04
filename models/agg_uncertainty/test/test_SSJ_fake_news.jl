@testset "Fake news algorithm (SSJ_fake_news.jl)" begin
    prim, res = steady_state()
    k′_ss = Linear.savings_on_hist(prim, res.k_policy)
    lottery = Linear.Lottery(prim.k_hist, k′_ss)

    @testset "steady-state transition" begin
        # The lottery preserves savings in expectation, and taking expectations with it is multiplication by Λₛₛ = T_star'
        @test Linear.expectation_step(lottery, prim.M, repeat(prim.k_hist, 1, prim.nz)) ≈ k′_ss
        x = rand(MersenneTwister(1), prim.n_hist, prim.nz)
        @test vec(Linear.expectation_step(lottery, prim.M, x)) ≈ res.T_star' * vec(x)

        # A shift in savings moves mass within each employment state, so the employment distribution is unchanged, and
        # raises capital tomorrow by the average shift
        dk′ = rand(MersenneTwister(2), prim.n_hist, prim.nz)
        dk′[k′_ss .>= prim.k_hist[end]] .= 0  # savings at the top of the grid stay there
        dD = Linear.forward_shock(lottery, prim.k_hist, prim.M, res.μ, dk′)
        @test maximum(abs.(sum(dD; dims=1))) < 1e-15
        @test sum(prim.k_hist .* dD) ≈ sum(res.μ .* dk′)
        # It is the derivative of the transition matrix: the lottery is linear in k′ between grid points
        ε = 1e-7
        res_shifted = deepcopy(res)
        res_shifted.k_pol_hist = k′_ss .+ ε .* dk′
        @test vec(dD) ≈ (Linear.build_T_star(prim, res_shifted) - res.T_star) * vec(res.μ) ./ ε rtol=1e-6
    end

    @testset "Jacobians" begin
        T = 60
        J = Linear.fake_news_jacobians(prim, res, T)
        # Capital at date 0 is predetermined
        @test all(iszero, J.K.r[1, :]) && all(iszero, J.K.w[1, :])

        # Proposition 1: the fake news algorithm gives the Jacobians of the direct method. SSJ2.jl's direct method takes
        # one-sided differences against an unperturbed path that drifts slightly from the steady state, an error of order
        # 1/h (0.02% of J_K_w for h = 1e-5). Averaging steps of ±h gives a central difference, as the fake news
        # algorithm takes, which removes the drift.
        J_K_r₊, J_K_w₊ = quietly(() -> Linear.fast_jacobian(prim, res, 1e-5, 1e-5, T))
        J_K_r₋, J_K_w₋ = quietly(() -> Linear.fast_jacobian(prim, res, -1e-5, -1e-5, T))
        @test J.K.r ≈ (J_K_r₊ + J_K_r₋) / 2 rtol=1e-5
        @test J.K.w ≈ (J_K_w₊ + J_K_w₋) / 2 rtol=1e-5

        # Central differences leave errors of order h² and rounding, so the step size hardly matters
        J_coarse = Linear.fake_news_jacobians(prim, res, T; h=1e-4)
        @test all(J_coarse[o][i] ≈ J[o][i] for o in (:K, :C), i in (:r, :w))

        # Household budgets C_t + K_{t+1} = (1 + r_t - δ)K_t + w_t L hold to first order for every date of the shock.
        # K and C come from different expectation vectors, so this checks the fake news matrices against each other.
        R = 1 + res.r - prim.δ
        L = sum(prim.ē .* prim.z_grid' .* res.μ)  # employment, which prices do not move
        same_date = Matrix(1.0I, T, T)[1:T-1, :]
        @test maxdiff(J.C.r[1:T-1, :] + J.K.r[2:T, :], R * J.K.r[1:T-1, :] + res.K * same_date) < 1e-9
        @test maxdiff(J.C.w[1:T-1, :] + J.K.w[2:T, :], R * J.K.w[1:T-1, :] + L * same_date) < 1e-9

        # Most of a surprise wage increase is saved: the employed, who receive it, are rarely constrained
        @test 0 < J.C.w[1, 1] / L < 0.1
    end

    @testset "general equilibrium" begin
        irf = Linear.SolveFakeNews_IRFs(prim, res; T=HORIZON, Z_shock=SHOCK)
        @test irf.dZ ≈ SHOCK .* 0.75 .^ (0:HORIZON-1)

        # The capital response matches SSJ2.jl's, up to its one-sided differences, and Reiter's
        window = 1:60
        ssj_K = ssj().irf_K
        @test maxdiff(irf.irf_K, ssj_K) < 1e-4 * maximum(ssj_K)
        @test maxdiff(irf.irf_K[window], reiter().irfs.K[window]) < 5e-4 * maximum(ssj_K)
        # Reiter computes consumption from the firm's resource constraint rather than households' budgets, so the two
        # differ by the steady state's capital market clearing error (as BKM does in test_methods_agree.jl)
        @test maxdiff(irf.irf_C[window], reiter().irfs.C[window]) < 0.01 * maximum(irf.irf_C)

        # Higher TFP raises both prices on impact, and consumption rises
        @test irf.dr[1] > 0 && irf.dw[1] > 0
        @test all(irf.irf_C[1:100] .> 0)
        # With constant returns, factor payments K dr + L dw add up to the change in output other than r dK, so
        # households' budgets aggregate to the resource constraint dC_t + dK_{t+1} = (1 + r - δ) dK_t + Y dZ_t
        Y = res.K^prim.α * prim.L^(1 - prim.α)
        R = 1 + res.r - prim.δ
        @test maxdiff(irf.dC[1:end-1] + irf.dK[2:end], R * irf.dK[1:end-1] + Y * irf.dZ[1:end-1]) < 1e-12
    end
end
