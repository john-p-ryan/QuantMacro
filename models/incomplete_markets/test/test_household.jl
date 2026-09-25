@testset "household problem" begin
    for (name, m) in METHODS
        @testset "$name" begin
            prim, res = baseline_household(m)
            k, c, k_next = prim.k_grid, res.c_policy, res.k_policy

            @testset "budget and borrowing constraints" begin
                @test maxdiff(c .+ k_next, cash_on_hand(prim, res)) < 1e-10
                @test all(c .> 0)
                @test all(prim.k_min .<= k_next .<= prim.k_max)
            end

            @testset "policy shape" begin
                # Saving and consumption rise with wealth, and the employed consume and save more than the unemployed.
                @test all(diff(k_next; dims=1) .>= 0)
                @test all(diff(c; dims=1) .> 0)
                @test all(c[:, 1] .> c[:, 2])
                @test all(k_next[:, 1] .>= k_next[:, 2])
            end

            @testset "precautionary and buffer-stock saving" begin
                # With β(1 + r - δ) < 1, a household with riskless income would stay at the borrowing constraint.
                # The employed save away from it only to self-insure against unemployment.
                @test k_next[1, 1] > prim.k_min + 0.1
                # Impatience wins at high wealth: the unemployed always dissave, and so do the employed at the top.
                # The unemployed's dissaving is tiny at low wealth, so on grid search's grid it can round to k' = k.
                @test all(k_next[2:end, 2] .<= k[2:end])
                @test k_next[end, 1] < k[end]
            end

            @testset "converged to a fixed point" begin
                @test maxdiff(updated_consumption(prim, res), c) <= policy_tol(prim)
            end

            @testset "independent of the starting point" begin
                # steady_state_capital! reuses one Results across candidate interest rates, so the solution must not
                # depend on where the iteration starts. At r = 0.06 > ρ + δ, households save up to the top of the grid.
                warm = solve_at_prices(prim, 0.06)
                solve_at_prices(prim, R_FIXED; res=warm)
                @test maxdiff(warm.k_policy, k_next) <= policy_tol(prim)
            end
        end
    end

    @testset "Euler equation" begin
        # Interpolated VFI's value function is inaccurate near the constraint, where V is steeply curved, so both
        # methods are compared away from it. EGM imposes the Euler equation at every endogenous grid point, so it
        # also holds, more loosely, near the constraint.
        for (m, tol) in ((EGM, 1e-4), (InterpVFI, 1e-3))
            prim, res = baseline_household(m)
            @test nanmax(euler_errors(prim, res)[prim.k_grid .>= 1, :]) < tol
        end
        @test nanmax(euler_errors(baseline_household(EGM)...)) < 1e-2
    end

    @testset "methods agree" begin
        prim_e, egm = baseline_household(EGM)
        prim_v, vfi = baseline_household(InterpVFI)
        prim_g, gs = baseline_household(GridSearch)
        @test prim_e.k_grid == prim_v.k_grid
        away = prim_e.k_grid .>= 1

        # Consumption within 0.01, about 1% of aggregate consumption.
        @test maxdiff(egm.c_policy[away, :], vfi.c_policy[away, :]) < 1e-2
        # Values within 0.25% of permanent consumption.
        quietly(() -> EGM.recover_V!(prim_e, egm; tol=1e-12))
        @test consumption_equivalent(prim_e, maxdiff(egm.V[away, :], vfi.V[away, :])) < 0.0025

        # Grid search picks k' on its uniform grid, so it matches EGM's savings to about a grid step.
        Δ = prim_g.k_grid[2] - prim_g.k_grid[1]
        k_next_egm = reduce(hcat, [evaluate_spline(egm.k_splines[z], prim_g.k_grid) for z in 1:prim_g.nz])
        @test maxdiff(k_next_egm, gs.k_policy) < 2Δ
    end

    @testset "closed form without labor income" begin
        # With no labor income, CRRA consumption is proportional to wealth and k' = (β(1 + r - δ))^(1/γ) k. The
        # borrowing constraint binds only at k_min.
        for γ in [1.0, 2.0], (name, m) in METHODS
            prim = m.Primitives(γ=γ, z_grid=[0.0, 0.0])
            res = solve_at_prices(prim, R_FIXED)
            k_exact = max.((prim.β * gross_return(prim, res))^(1 / γ) .* prim.k_grid, prim.k_min)
            tol, region = closed_form_tol(prim)
            @test maxdiff(res.k_policy[region, :], k_exact[region]) <= tol
        end

        @testset "EGM value function with log utility" begin
            prim = EGM.Primitives(z_grid=[0.0, 0.0])
            res = solve_at_prices(prim, R_FIXED)
            quietly(() -> EGM.recover_V!(prim, res; tol=1e-12))
            β, R = prim.β, gross_return(prim, res)
            # Guess and verify V(k) = a + log(k) / (1 - β) in V(k) = log((1 - β)Rk) + βV(βRk).
            a = (log((1 - β) * R) + β / (1 - β) * log(β * R)) / (1 - β)
            away = prim.k_grid .>= 1
            V_exact = a .+ log.(prim.k_grid[away]) ./ (1 - β)
            @test consumption_equivalent(prim, maxdiff(res.V[away, :], V_exact)) < 5e-4
        end
    end
end
