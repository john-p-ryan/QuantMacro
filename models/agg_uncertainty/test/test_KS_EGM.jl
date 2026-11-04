@testset "Krusell-Smith, EGM and histogram (KS_EGM_hist.jl)" begin
    prim_a, res_a = ks_aiyagari()

    @testset "primitives" begin
        prim = KSEGM.Primitives()
        # Symmetric two-state TFP chain with mean 1 and autocorrelation ρ, the process the linear methods use
        @test all(sum(prim.Mzz; dims=2) .≈ 1) && prim.Mzz == prim.Mzz'
        @test prim.Mzz[1, 1] - prim.Mzz[2, 1] ≈ prim.ρ
        @test stationary(prim.Mzz) ≈ [0.5, 0.5]
        @test (prim.z_g + prim.z_b) / 2 ≈ 1
        @test KSEGM.Primitives(; ρ=0.5).Mzz[1, 1] ≈ 0.75
        # Unemployment and labor are constant: the employment chain does not depend on z
        @test stationary(prim.M_unemp)[2] ≈ prim.unemp
        @test prim.L ≈ prim.ē * (1 - prim.unemp)
        # Grids span their bounds, and the histogram grid is the Aiyagari steady state's
        @test prim.k_grid[1] ≈ prim.k_min && prim.k_grid[end] ≈ prim.k_max && issorted(prim.k_grid)
        @test prim.k_hist_grid == prim_a.k_hist
        @test_throws AssertionError KSEGM.Primitives(; ρ=1.0)
        @test_throws AssertionError KSEGM.Primitives(; γ=0.0)
        @test_throws ArgumentError KSEGM.Initialize(; not_a_parameter=1.0)
    end

    @testset "marginal utility" begin
        for γ in (1.0, 2.0), c in (0.01, 0.5, 3.0)
            @test KSEGM.u_prime_inv(KSEGM.u_prime(c, γ), γ) ≈ c
        end
        # Below ε = 1e-6 a first-order Taylor expansion keeps u' continuous, decreasing, and finite at c ≤ 0
        ε = 1e-6
        for γ in (1.0, 2.0)
            @test KSEGM.u_prime(ε * (1 - 1e-9), γ) ≈ KSEGM.u_prime(ε * (1 + 1e-9), γ) rtol=1e-6
            @test isfinite(KSEGM.u_prime(-1.0, γ))
            @test KSEGM.u_prime(-1.0, γ) > KSEGM.u_prime(0.0, γ) > KSEGM.u_prime(ε, γ)
        end
    end

    @testset "initialization from the Aiyagari steady state" begin
        prim, sim, res, μ0 = quietly(() -> KSEGM.Initialize(; T=10))
        @test μ0 ≈ res_a.μ
        @test all(μ0 .>= 0) && sum(μ0) ≈ 1
        @test sum(μ0 .* prim.k_hist_grid) ≈ res_a.K
        # The initial law of motion has its fixed point at the Aiyagari capital stock in both states
        @test exp(res.a₀ / (1 - res.a₁)) ≈ res_a.K && exp(res.b₀ / (1 - res.b₁)) ≈ res_a.K
        # The initial policy consumes all cash on hand
        @test res.c_policy ≈ ks_cash_on_hand(prim)
    end

    @testset "without aggregate risk the household problem is Aiyagari's" begin
        # With z ≡ 1 and the law of motion K' = K, a household at the K grid point equal to the capital demanded at the
        # Aiyagari interest rate faces that model's constant prices, so it should choose the same policies.
        K_ss = capital_demand(prim_a, res_a.r)
        prim, sim, res, μ0 = quietly(() -> KSEGM.Initialize(; z_grid=[1.0, 1.0], K_min=K_ss - 1, K_max=K_ss + 1,
                                                              nK=3, T=400, burn=100))
        @test prim.K_grid[2] ≈ K_ss
        res.a₀, res.a₁, res.b₀, res.b₁ = 0.0, 1.0, 0.0, 1.0
        res = quietly(() -> KSEGM.Policy_Iteration_EGM(prim, res, sim))
        for z in 1:2
            @test maxdiff(res.c_policy[:, :, 2, z], res_a.c_policy) < 1e-6
            @test maxdiff(res.k_policy[:, :, 2, z], res_a.k_policy) < 1e-6
        end

        # Starting from the Aiyagari distribution, the simulated economy stays at its steady state. It drifts only by
        # the steady state's own market-clearing error and the difference between the linear interpolation of the
        # savings policy here and the PCHIP interpolation in the Aiyagari solver.
        K, C, r, w, K_var, Z = quietly(() -> KSEGM.SimulateDistributionPath(prim, res, sim, μ0))
        @test all(Z .== 1)
        @test K[1] ≈ res_a.K
        @test maximum(abs.(K ./ res_a.K .- 1)) < 1e-4
        # KS aggregates consumption with linear interpolation in k and the Aiyagari solver with PCHIP, which differ by
        # about 5e-5 on the same policy and distribution
        @test C[1] ≈ res_a.C rtol=1e-4
        @test K_var[1] ≈ res_a.K_var
        @test r ≈ rental_rate.(Ref(prim), K) && w ≈ wage.(Ref(prim), K)
    end

    @testset "household problem with aggregate risk" begin
        (; prim, res) = ks_egm()
        k_next, c = res.k_policy, res.c_policy
        @test maxdiff(c .+ k_next, ks_cash_on_hand(prim)) < 1e-10
        @test all(c .> 0) && all(k_next .>= prim.k_min)
        # Saving and consumption rise with wealth; the employed and households in good times consume more
        @test all(diff(k_next; dims=1) .>= 0) && all(diff(c; dims=1) .> 0)
        @test all(c[:, 1, :, :] .> c[:, 2, :, :])
        @test all(c[:, :, :, 1] .> c[:, :, :, 2])
        # A fixed point of the EGM operator. SolveModel updates the law of motion once more after its last policy
        # solve, so the policy is the fixed point for a law of motion that differs from res's by less than reg_tol.
        _, c_more = KSEGM.Bellman_EGM(prim, res)
        @test maxdiff(c_more, c) < 1e-6
        # Euler equation: tight away from the constraint, and within 1% where the policy's kink is hard to interpolate
        errors = ks_euler_errors(prim, res)
        @test nanmax(errors[prim.k_grid .>= 1, :, :, :]) < 1e-4
        @test nanmax(errors) < 1e-2
    end

    @testset "simulation" begin
        (; prim, sim, res) = ks_egm()
        K, C, r, w, K_var, Z = res.K_sim_path, res.C_sim_path, res.r_sim_path, res.w_sim_path, res.K_var_sim_path, res.Z_sim_path
        @test all(in(prim.z_grid), Z) && Z[1] == prim.z_g
        @test r ≈ rental_rate.(Ref(prim), K, Z) && w ≈ wage.(Ref(prim), K, Z)
        @test all(K_var .> 0)
        # Aggregating budgets over the histogram gives the goods market identity C + K' - (1 - δ)K = Y = rK + wL, up to
        # the bilinear interpolation of the policies in K
        Y = r .* K .+ w .* prim.L
        @test maximum(abs.((C[1:end-1] .+ K[2:end] .- (1 - prim.δ) .* K[1:end-1]) ./ Y[1:end-1] .- 1)) < 2e-4

        # One step of the histogram method maps μ to the mean of the savings policy: the lottery over adjacent grid
        # points preserves each household's k'
        K2, = quietly(() -> KSEGM.SimulateDistributionPath(prim, res, KSEGM.Simulations(; T=2), res_a.μ))
        k_itps = [BilinearSpline(prim.k_grid, prim.K_grid, res.k_policy[:, ϵ, :, 1]; bc_type="constant") for ϵ in 1:prim.nϵ]
        n = prim.n_hist
        K_next = sum(res_a.μ[:, ϵ] ⋅ evaluate_spline(k_itps[ϵ], prim.k_hist_grid, fill(K2[1], n)) for ϵ in 1:prim.nϵ)
        @test K2[2] ≈ K_next rtol=1e-8  # up to the mass below 1e-12 that the histogram drops
    end

    @testset "law of motion" begin
        (; prim, sim, res) = ks_egm()
        # Converged: re-estimating on the final simulation reproduces the law of motion households used
        a₀, a₁, b₀, b₁, R² = quietly(() -> KSEGM.EstimateRegression(prim, res.K_sim_path, res.Z_sim_path, sim))
        @test maxdiff([a₀, a₁, b₀, b₁], [res.a₀, res.a₁, res.b₀, res.b₁]) < 10 * sim.reg_tol
        @test R² ≈ res.R²
        # Approximate aggregation: with 1% TFP shocks the mean of capital is a nearly sufficient statistic
        @test res.R² > 0.9999
        # Stable and persistent in both states; capital is pulled up in good times and down in bad times
        @test 0.9 < res.a₁ < 1 && 0.9 < res.b₁ < 1
        K_sim = res.K_sim_path[(sim.burn+1):end]
        K_good, K_bad = exp(res.a₀ / (1 - res.a₁)), exp(res.b₀ / (1 - res.b₁))
        @test K_bad < mean(K_sim) < K_good
        @test prim.K_min <= minimum(K_sim) && maximum(K_sim) <= prim.K_max  # no extrapolation off the K grid
        # Small aggregate risk barely moves average capital away from the Aiyagari steady state
        @test mean(K_sim) ≈ res_a.K rtol=0.01

        # The regression recovers a law of motion that generated a path exactly
        lom = (a₀=0.09, a₁=0.962, b₀=0.084, b₁=0.966)
        K = [11.5]
        for t in 1:sim.T - 1
            push!(K, res.Z_sim_path[t] == prim.z_g ? exp(lom.a₀ + lom.a₁ * log(K[t])) : exp(lom.b₀ + lom.b₁ * log(K[t])))
        end
        estimate = quietly(() -> KSEGM.EstimateRegression(prim, K, res.Z_sim_path, sim))
        @test collect(estimate[1:4]) ≈ collect(lom) rtol=1e-8
        @test estimate[5] ≈ 1
    end

    @testset "generalized impulse response" begin
        (; prim, sim, res) = ks_egm()
        # When the law of motion ignores z, the shock cannot move capital. The raw path after a good shock would still
        # drift, since the long-run mean of capital is not this law of motion's fixed point; the baseline removes it.
        flat = deepcopy(res)
        flat.b₀, flat.b₁ = flat.a₀, flat.a₁
        @test exp(flat.a₀ / (1 - flat.a₁)) / mean(res.K_sim_path[(sim.burn+1):end]) - 1 > 1e-3
        irf_K, _ = KSEGM.estimate_impulse_response(prim, flat, sim; num_simulations=100, max_horizon=30)
        @test all(irf_K .== 0)

        # Mean TFP after a good shock decays at rate ρ toward its unconditional mean of 1
        Random.seed!(7)
        irf_K, irf_z = KSEGM.estimate_impulse_response(prim, res, sim; num_simulations=100_000, max_horizon=40)
        @test irf_z ≈ 1 .+ (prim.z_g - 1) .* prim.ρ .^ (0:39) atol=3e-4
        # Capital rises in a hump and is still above the baseline after 40 quarters
        peak = argmax(irf_K)
        @test all(irf_K .> 0) && 3 < peak < 20 && irf_K[end] < irf_K[peak]
    end
end
