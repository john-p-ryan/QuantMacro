@testset "BKM transition path (BKM_transition.jl)" begin
    prim, res = steady_state()

    @testset "MIT shock" begin
        sim = Linear.Simulation(; T=50, Z_shock=0.02, ρ=0.8)
        tr = Linear.InitializeTransition(prim, res, sim)
        # log Z decays deterministically at rate ρ from the initial shock
        @test tr.Z_path ≈ (1 + 0.02) .^ (0.8 .^ (0:50))
        @test tr.K_path == fill(res.K, 51)
        @test tr.r_path ≈ rental_rate.(Ref(prim), tr.K_path, tr.Z_path)
        @test tr.w_path ≈ wage.(Ref(prim), tr.K_path, tr.Z_path)
        @test tr.μ_path[:, :, 1] == res.μ && tr.k_policy_path[:, :, end] == res.k_policy
    end

    @testset "household block at steady-state prices" begin
        sim = Linear.Simulation(; T=5)
        tr = Linear.InitializeTransition(prim, res, sim)
        tr.r_path .= res.r
        tr.w_path .= res.w
        # One backward step is the Aiyagari solver's EGM step, and the steady-state policy is its fixed point
        k_next, c_next = Linear.BellmanTransition(prim, tr, sim, sim.T)
        k_a, c_a = Linear.bellman(prim, res)
        @test k_next ≈ k_a && c_next ≈ c_a
        @test maxdiff(k_next, res.k_policy) < 1e-10

        # Iterating forward with the steady-state policy leaves the stationary distribution in place
        Linear.BackwardIteration!(prim, tr, sim)
        Linear.ForwardIteration!(prim, tr, sim)
        @test maxdiff(tr.k_pol_hist_path[:, :, 1], res.k_pol_hist) < 1e-10
        @test vec(tr.μ_path[:, :, 2]) ≈ res.T_star * vec(res.μ)  # same lottery as the steady state's transition matrix
        @test all(maxdiff(tr.μ_path[:, :, t], res.μ) < 1e-10 for t in 1:sim.T+1)
    end

    @testset "no shock" begin
        sim, tr = quietly(() -> Linear.SolveTransition_from_SS(prim, res; T=60, Z_shock=0.0))
        # Nothing moves, except by the steady state's own market-clearing error
        @test maximum(abs.(tr.K_path ./ res.K .- 1)) < max(2 * market_gap(), 1e-8)
        @test maxdiff(tr.k_policy_path[:, :, 1], res.k_policy) < 1e-3
    end

    @testset "1% TFP shock" begin
        (; sim, tr, irfs) = bkm()
        T = sim.T

        @testset "equilibrium path" begin
            # Converged: households' capital supply equals the path that set prices
            @test maxdiff(Linear.AggregateCapital(sim, tr)[2:T], tr.K_path[2:T]) < 1e-9
            @test tr.r_path ≈ rental_rate.(Ref(prim), tr.K_path, tr.Z_path)
            @test tr.Y_path ≈ tr.Z_path .* tr.K_path .^ prim.α .* prim.L^(1 - prim.α)
            # The histogram lottery preserves mass and each household's k', except that the few households at the top of
            # the grid who would save beyond k_max after the shock are held at k_max
            for t in (1, 10, T)
                μ, μ_next = tr.μ_path[:, :, t], tr.μ_path[:, :, t+1]
                k_next = min.(tr.k_pol_hist_path[:, :, t], prim.k_max)
                @test sum(μ_next) ≈ 1
                @test sum(μ_next .* prim.k_hist) ≈ sum(μ .* k_next) rtol=1e-12
            end
            # Aggregating budgets gives the goods market identity C + K' - (1 - δ)K = Y, up to PCHIP interpolation of the
            # policies on the histogram grid
            gap = (tr.C_path[1:T] .+ tr.K_path[2:T+1] .- (1 - prim.δ) .* tr.K_path[1:T]) ./ tr.Y_path[1:T] .- 1
            @test maximum(abs.(gap)) < 1e-3
        end

        @testset "impulse responses" begin
            @test irfs.Z ≈ ((1 + SHOCK) .^ (sim.ρ .^ (0:T)) .- 1) ./ SHOCK
            # Capital is predetermined on impact, then rises in a hump and returns to the steady state
            K = irfs.K
            peak = argmax(K)
            @test K[1] == 0
            @test all(K[2:100] .> 0) && 5 < peak < 30
            @test abs(K[end-1]) < 0.05 * K[peak]
            # On impact prices and output move one for one with TFP; consumption rises by less (smoothing)
            @test irfs.r[1] ≈ 1 atol=0.01
            @test irfs.w[1] ≈ 1 atol=0.01
            @test irfs.Y[1] ≈ 1 atol=0.01
            @test 0 < irfs.C[1] < irfs.Y[1]
            # Once TFP has mostly decayed, the extra capital depresses the return to capital and raises the wage
            @test irfs.r[peak] < 0 < irfs.w[peak]
        end

        @testset "simulation by convolution" begin
            # Feeding the MIT shock's own TFP path back in reproduces the transition, in percent and in level IRFs
            path = Linear.simulate_economy(tr.Z_path, irfs, res, prim, sim)
            @test path.K ≈ tr.K_path && path.C ≈ tr.C_path && path.r ≈ tr.r_path
            levels = Linear.ComputeIRFs(prim, res, tr, sim; use_levels=true)
            path = Linear.simulate_economy(tr.Z_path, levels, res, prim, sim; use_levels=true)
            @test path.K ≈ tr.K_path && path.C ≈ tr.C_path && path.r ≈ tr.r_path
            # With tfp_levels the innovations come from Z - 1, so a single shock is a path with Z - 1 decaying at rate ρ
            Z_lev = 1 .+ log(1 + sim.Z_shock) .* sim.ρ .^ (0:T)
            path = Linear.simulate_economy(Z_lev, levels, res, prim, sim; use_levels=true, tfp_levels=true)
            @test path.K ≈ tr.K_path && path.C ≈ tr.C_path && path.r ≈ tr.r_path
            Z_lev = 1 .+ sim.Z_shock .* sim.ρ .^ (0:T)
            path = Linear.simulate_economy(Z_lev, irfs, res, prim, sim; tfp_levels=true)
            @test path.K ≈ tr.K_path && path.C ≈ tr.C_path && path.r ≈ tr.r_path

            @test Linear.convolve([1.0, 0.5, 0.25], [1.0, 0.0, 0.0, 0.0]) == [1.0, 0.5, 0.25, 0.0]
            @test Linear.convolve([1.0, 0.5, 0.25], [0.0, 2.0, 0.0, 1.0]) == [0.0, 2.0, 1.0, 1.5]
        end
    end
end
