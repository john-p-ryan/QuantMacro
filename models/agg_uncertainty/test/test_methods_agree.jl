@testset "methods agree" begin
    window = 1:60  # quarters after the shock, covering the hump in capital

    # Capital IRFs to a 1% TFP shock in percent deviations per percent shock, with the shock period first
    bkm_K = bkm().irfs.K[1:HORIZON]
    reiter_K = reiter().irfs.K[1:HORIZON]
    ssj_K = ssj().irf_K
    peak = maximum(ssj_K)

    @testset "linear methods" begin
        # Reiter and SSJ are both first order around the same steady state and the same TFP path, so they differ only
        # by SSJ's finite-difference Jacobians and Reiter's treatment of the borrowing constraint
        @test maxdiff(reiter_K[window], ssj_K[window]) < 5e-4 * peak
        # BKM solves the nonlinear perfect-foresight path, so it also differs by terms of the order of the shock, and it
        # inherits the steady state's market-clearing error divided by the shock size (≈ 0.3% of K, 1% of the peak)
        @test market_gap() / SHOCK < 0.01
        @test maxdiff(bkm_K[window], ssj_K[window]) < 0.02 * peak
        @test maxdiff(bkm_K[window], reiter_K[window]) < 0.02 * peak
        @test argmax(bkm_K) == argmax(reiter_K) == argmax(ssj_K)
    end

    @testset "Krusell-Smith global solution" begin
        # KS discretizes TFP as a symmetric two-state chain with the linear methods' persistence and a 1% deviation
        # from its mean, so the expected path of TFP after a good shock matches the MIT shock to first order. With
        # shocks this small the global solution is nearly linear (approximate aggregation), so its generalized IRF
        # matches the first-order one up to nonlinear terms and sampling error.
        (; prim, sim, res) = ks_egm()
        @test prim.z_g - 1 ≈ SHOCK && prim.ρ == 0.75
        ks_K = ks_irf(prim, res, sim)
        @test ks_K[1] == 0
        @test maxdiff(ks_K[window], ssj_K[window]) < 0.01 * peak
        @test abs(argmax(ks_K) - argmax(ssj_K)) <= 2
    end

    @testset "simulated paths" begin
        # Driving the linear methods with KS's simulated TFP path reproduces KS's aggregate capital closely
        prim_ss, res_ss = steady_state()
        ks = ks_egm()
        burn = (ks.sim.burn+1):ks.sim.T
        Z = ks.res.Z_sim_path
        bkm_path = Linear.simulate_economy(Z, bkm().irfs, res_ss, prim_ss, bkm().sim)
        (; system, g_x, h_x) = reiter()
        reiter_path = Linear.simulate_reiter_economy(Z, system, g_x, h_x, res_ss, prim_ss)
        K_ks = ks.res.K_sim_path[burn]
        for K_linear in (bkm_path.K[burn], reiter_path.K[burn])
            @test cor(K_linear, K_ks) > 0.98
            @test std(K_linear) ≈ std(K_ks) rtol=0.1
            @test mean(K_linear) ≈ mean(K_ks) rtol=0.01
        end
    end
end
