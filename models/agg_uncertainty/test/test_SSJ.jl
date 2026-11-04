@testset "Sequence-space Jacobian (SSJ2.jl)" begin
    prim, res = steady_state()

    @testset "backward step" begin
        # At steady-state prices and marginal value ∂V = (1 + r - δ) u'(c), one EGM step returns the steady state
        ∂V = (1 + res.r - prim.δ) .* Linear.u_prime.(res.c_policy, prim.γ)
        k_next, c, ∂V_new = Linear.backward_iterate(prim, ∂V, res.r, res.w)
        @test maxdiff(k_next, res.k_policy) < 1e-10 && maxdiff(∂V_new, ∂V) < 1e-9
        @test c .+ k_next ≈ (1 + res.r - prim.δ) .* prim.k_grid .+ res.w * prim.ē .* prim.z_grid'

        # Without a price change every transition matrix in the sequence is the steady state's
        T_shock, T_seq = Linear.compute_policy_sequences(prim, res, 5)
        @test maximum(abs.(T_shock .- res.T_star)) < 1e-10
        @test all(maximum(abs.(T_j .- res.T_star)) < 1e-10 for T_j in T_seq)
        @test all(sum(T_shock; dims=1) .≈ 1)  # columns are distributions over tomorrow's states
    end

    @testset "firm block" begin
        T = 4
        J_r_K, J_w_K, J_r_Z, J_w_Z = Linear.Firm_Block(prim, res, T)
        h = 1e-6
        K = res.K
        diagonal(x) = Matrix(x * I, T, T)  # prices depend only on same-period capital and TFP
        @test J_r_K ≈ diagonal((rental_rate(prim, K + h) - rental_rate(prim, K - h)) / 2h) rtol=1e-6
        @test J_w_K ≈ diagonal((wage(prim, K + h) - wage(prim, K - h)) / 2h) rtol=1e-6
        @test J_r_Z ≈ diagonal((rental_rate(prim, K, 1 + h) - rental_rate(prim, K, 1 - h)) / 2h) rtol=1e-6
        @test J_w_Z ≈ diagonal((wage(prim, K, 1 + h) - wage(prim, K, 1 - h)) / 2h) rtol=1e-6
    end

    @testset "household Jacobians" begin
        T = 40
        J_K_r, J_K_w = quietly(() -> Linear.fast_jacobian(prim, res, 1e-5, 1e-5, T))
        # Capital in period 1, when the shock is revealed, is predetermined
        @test all(iszero, J_K_r[1, :]) && all(iszero, J_K_w[1, :])

        # Brute force: perturb one period's price in BKM's backward-forward iteration and difference the capital path
        sim = Linear.Simulation(; T)
        function capital_path(; r_at=0, w_at=0, h=1e-5)
            tr = Linear.InitializeTransition(prim, res, sim)
            tr.r_path .= res.r
            tr.w_path .= res.w
            r_at > 0 && (tr.r_path[r_at] += h)
            w_at > 0 && (tr.w_path[w_at] += h)
            Linear.BackwardIteration!(prim, tr, sim)
            Linear.ForwardIteration!(prim, tr, sim)
            return Linear.AggregateCapital(sim, tr)[1:T]
        end
        # Both use the same one-sided step, so they agree up to rounding amplified by 1/h: SSJ's shocked paths are built
        # from steady-state policies re-derived by backward iteration, which differ from res's by ~1e-12. A misaligned
        # timing convention would instead give differences of order one.
        baseline = capital_path()
        for s in (1, 5, 20)
            @test J_K_r[:, s] ≈ (capital_path(r_at=s) .- baseline) ./ 1e-5 rtol=1e-4
            @test J_K_w[:, s] ≈ (capital_path(w_at=s) .- baseline) ./ 1e-5 rtol=1e-4
        end

        # A higher return in period s raises saving before it (substitution) and after it (income)
        @test all(J_K_r[2:end, :] .> 0)
        # An anticipated higher wage lowers saving before it is paid (smoothing) and raises it after
        s = 20
        @test all(J_K_w[2:s, s] .< 0) && all(J_K_w[s+1:end, s] .> 0)
    end

    @testset "general equilibrium" begin
        (; J_K_r, J_K_w, J_r_K, J_w_K, J_r_Z, J_w_Z, dZ, dK, irf_K) = ssj()
        @test dZ ≈ SHOCK .* 0.75 .^ (0:HORIZON-1)  # first-order TFP path
        # The solution is linear, so the response per unit shock does not depend on the shock's size
        _, dK_big = Linear.solve_dK(0.75, 10SHOCK, J_K_r, J_K_w, J_r_K, J_w_K, J_r_Z, J_w_Z)
        @test dK_big ./ 10SHOCK ≈ dK ./ SHOCK
        # dK solves the linearized capital market clearing condition H_K dK + H_Z dZ = 0
        H_K = J_K_r * J_r_K + J_K_w * J_w_K - I
        H_Z = J_K_r * J_r_Z + J_K_w * J_w_Z
        @test maximum(abs.(H_K * dK + H_Z * dZ)) < 1e-10
        # Capital is predetermined on impact, then rises in a hump and returns to the steady state
        peak = argmax(irf_K)
        @test dK[1] == 0
        @test all(irf_K[2:100] .> 0) && 5 < peak < 30
        @test abs(irf_K[end]) < 0.05 * irf_K[peak]

        # The wrapper runs the same steps
        T = 30
        dZ_w, dK_w, dK_dZ_w = quietly(() -> Linear.SolveSSJ_IRFs(prim, res; T, Z_shock=SHOCK))
        J_K_r, J_K_w = quietly(() -> Linear.fast_jacobian(prim, res, 1e-5, 1e-5, T))
        @test dK_w ≈ Linear.solve_dK(0.75, SHOCK, J_K_r, J_K_w, Linear.Firm_Block(prim, res, T)...)[2]
        @test dK_dZ_w ≈ dK_w ./ (SHOCK * res.K)
    end
end
