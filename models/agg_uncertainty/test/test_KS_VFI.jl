@testset "Krusell-Smith, VFI (KS.jl)" begin
    @testset "shock process" begin
        # The default has equal durations of good and bad times. With unequal ones the joint chain must still use the
        # aggregate transition probabilities in the right blocks.
        for sho in (KSVFI.Shocks(), KSVFI.Shocks(; d_b=4.0, u_b=0.12))
            @test all(sho.M .>= 0) && all(sum(sho.M; dims=2) .≈ 1)
            @test all(sho.MZZ .>= 0) && all(sum(sho.MZZ; dims=2) .≈ 1)
            # Summing out ϵ' gives the aggregate transition, whatever the household's employment status
            for z in 1:2, ϵ in 1:2, z′ in 1:2
                @test sum(sho.M[2(z - 1) + ϵ, 2(z′ - 1) .+ (1:2)]) ≈ sho.MZZ[z, z′]
            end
            # Employment transitions carry the unemployment rate of today's aggregate state into tomorrow's (KS 1998)
            u = [sho.u_g, sho.u_b]
            employment = [sho.Mgg, sho.Mgb, sho.Mbg, sho.Mbb]  # from z to z′, in the order (g, g), (g, b), (b, g), (b, b)
            for (E, (z, z′)) in zip(employment, ((1, 1), (1, 2), (2, 1), (2, 2)))
                @test [1 - u[z] u[z]] * E ≈ [1 - u[z′] u[z′]]
            end
            # Expected durations of unemployment spells and of aggregate states
            @test 1 / (1 - sho.pgg00) ≈ sho.d_ug && 1 / (1 - sho.pbb00) ≈ sho.d_ub
            @test 1 / (1 - sho.pgg) ≈ sho.d_g && 1 / (1 - sho.pbb) ≈ sho.d_b
        end
    end

    @testset "shock simulation" begin
        Random.seed!(42)
        P = [0.7 0.2 0.1; 0.1 0.6 0.3; 0.25 0.25 0.5]
        draws = [KSVFI.sim_markov(2, P) for _ in 1:100_000]
        @test [mean(draws .== j) for j in 1:3] ≈ P[2, :] atol=0.01

        prim, sho = KSVFI.Primitives(), KSVFI.Shocks()
        sim = KSVFI.Simulations(; T=2000, N=2000)
        Z, E = KSVFI.draw_shocks(prim, sho, sim)
        @test length(Z) == sim.T && size(E) == (sim.N, sim.T)
        @test Z[1] == prim.Z_g && all(in(prim.Z_grid), Z) && all(in((0.0, 1.0)), E)
        @test KSVFI.draw_shocks(prim, sho, sim) == (Z, E)  # reproducible from the seed
        # Law of large numbers: the cross-sectional unemployment rate tracks u_g in good times and u_b in bad times
        unemployment = vec(1 .- mean(E; dims=1))
        good = Z .== prim.Z_g
        @test mean(unemployment[good]) ≈ sho.u_g atol=2e-3
        @test mean(unemployment[.!good]) ≈ sho.u_b atol=2e-3
        @test mean(good) ≈ 0.5 atol=0.1
        # The initial number of unemployed is rounded, so any panel size works
        _, E_odd = KSVFI.draw_shocks(prim, sho, KSVFI.Simulations(; T=3, N=333))
        @test count(==(0.0), E_odd[:, 1]) == round(Int, 333 * sho.u_g)
    end

    @testset "utility" begin
        u = KSVFI.u
        for c in (0.3, 1.0, 4.0)
            @test u(c, 1.0) ≈ log(c)
            @test u(c, 2.0) ≈ 1 - 1 / c
            @test u(c, 1.0 + 1e-9) ≈ log(c) atol=1e-8  # the normalization makes u continuous in γ at log utility
        end
        # Below ε = 1e-4 utility is extended linearly with slope u'(ε), so it is continuous and increasing
        ε = 1e-4
        for γ in (1.0, 2.0)
            @test u(ε * (1 - 1e-9), γ) ≈ u(ε, γ) rtol=1e-8
            @test (u(ε / 2, γ) - u(ε / 4, γ)) / (ε / 4) ≈ ε^(-γ)
            @test u(-1.0, γ) < u(0.0, γ) < u(ε, γ)
        end
    end

    @testset "bilinear interpolation" begin
        x, y = collect(range(0.0, 10.0; length=11)), collect(range(5.0, 8.0; length=4))
        f(a, b) = 1 + 2a - 3b + 0.5a * b  # bilinear, so interpolation is exact
        itp = KSVFI.bilinear_interp([f(a, b) for a in x, b in y], x, y)
        @test itp(3.7, 6.2) ≈ f(3.7, 6.2)
        @test itp(12.0, 9.0) ≈ f(10.0, 8.0)  # flat extrapolation
        @test itp(-1.0, 6.5) ≈ f(0.0, 6.5)
    end

    # A coarse grid keeps value function iteration cheap
    prim_coarse = KSVFI.Primitives(; nk=20, nK=4)

    @testset "household problem" begin
        prim, sho, sim, res = KSVFI.initialize(; prim=prim_coarse, sim=KSVFI.Simulations(; N=500, T=600, burn=100))
        L = [prim.ē * (1 - sho.u_g), prim.ē * (1 - sho.u_b)]
        budget = [(1 + prim.α * z * (L[iz] / K)^(1 - prim.α) - prim.δ) * k + (1 - prim.α) * z * (K / L[iz])^prim.α * prim.ē * ϵ
                  for k in prim.k_grid, ϵ in prim.ϵ_grid, K in prim.K_grid, (iz, z) in enumerate(prim.Z_grid)]

        @testset "Bellman operator" begin
            # With no continuation value, the household consumes its whole budget
            V, k_next = KSVFI.bellman(prim, res, sho)
            @test maximum(k_next) < 1e-6
            @test V ≈ KSVFI.u.(budget, prim.γ) rtol=1e-8

            # With a continuation value that depends only on tomorrow's (ϵ', z'), saving is still worthless, and the
            # expectation must weight each (ϵ', z') by P(z' | z) P(ϵ' | ϵ, z, z'). Unequal durations of good and bad
            # times make the aggregate transition probabilities differ across rows.
            sho_unequal = KSVFI.Shocks(; d_b=4.0)
            v = [2.0 -1.0; 0.5 -3.0]  # v[ϵ', z']
            flat = deepcopy(res)
            for ϵ′ in 1:2, z′ in 1:2
                flat.V[:, ϵ′, :, z′] .= v[ϵ′, z′]
            end
            V, k_next = KSVFI.bellman(prim, flat, sho_unequal)
            @test maximum(k_next) < 1e-6
            employment = Dict((1, 1) => sho_unequal.Mgg, (1, 2) => sho_unequal.Mgb,
                              (2, 1) => sho_unequal.Mbg, (2, 2) => sho_unequal.Mbb)  # from z to z′
            for ϵ in 1:2, z in 1:2
                EV = sum(sho_unequal.MZZ[z, z′] * (employment[(z, z′)][ϵ, :] ⋅ v[:, z′]) for z′ in 1:2)
                @test V[:, ϵ, :, z] ≈ KSVFI.u.(budget[:, ϵ, :, z], prim.γ) .+ prim.β * EV rtol=1e-8
            end
        end

        @testset "value function iteration" begin
            sol = quietly(() -> KSVFI.vfi(prim, res, sho, sim))
            V, k_next = sol.V, sol.k_policy

            # Converged: one more application of the Bellman operator leaves V unchanged up to the relative tolerance
            V_more, _ = KSVFI.bellman(prim, sol, sho)
            @test maximum(abs.(V_more .- V) ./ (1 .+ abs.(V))) <= sim.V_tol

            @test all(0 .<= k_next .< budget)  # positive consumption
            # Value rises with wealth and is higher when employed
            @test all(diff(V; dims=1) .> 0)
            @test all(V[:, 1, :, :] .> V[:, 2, :, :])
            # Good times are better (higher return, lower unemployment risk). This is checked on the lower half of the
            # grid: at K_min, β(1 + r - δ) > 1, so wealthy households save toward k_max, where the flat extrapolation of
            # V cuts off the gains from saving, and more so in good times when r is higher.
            lower = prim.k_grid .<= prim.k_max / 2
            @test all(V[lower, :, :, 1] .> V[lower, :, :, 2])
            # Saving rises with wealth (up to Brent's tolerance) and the employed save more
            @test all(diff(k_next; dims=1) .>= -1e-6)
            @test all(k_next[:, 1, :, :] .>= k_next[:, 2, :, :] .- 1e-6)
            # Impatience: at K_max, where β(1 + r - δ) < 1 in both states, the wealthiest households dissave
            r_at_K_max = [prim.α * z * (L[iz] / prim.K_max)^(1 - prim.α) for (iz, z) in enumerate(prim.Z_grid)]
            @test all(prim.β .* (1 .+ r_at_K_max .- prim.δ) .< 1)
            @test all(k_next[end, :, end, :] .< prim.k_grid[end])
        end
    end

    @testset "simulation aggregates a linear policy exactly" begin
        # Bilinear interpolation is exact for a policy linear in k, so K' = a + bK holds for the cross-sectional mean
        prim, sho, sim, res = KSVFI.initialize(; prim=prim_coarse, sim=KSVFI.Simulations(; N=300, T=50))
        a, b = 1.2, 0.9
        res.k_policy .= a .+ b .* prim.k_grid
        K_path, k_panel = KSVFI.simulate_capital_path(prim, res, sim)
        K_exact = accumulate((K, _) -> a + b * K, 1:sim.T - 1; init=sim.K_initial)
        @test K_path ≈ [sim.K_initial; K_exact]
        @test k_panel[:, end] ≈ fill(K_exact[end], sim.N)
    end

    @testset "regression recovers a known law of motion" begin
        prim, sho, sim, res = KSVFI.initialize(; sim=KSVFI.Simulations(; N=10, T=2000, burn=100))
        lom = (a₀=0.095, a₁=0.962, b₀=0.085, b₁=0.965)
        K = [11.0]
        for t in 1:sim.T - 1
            push!(K, res.Z[t] == prim.Z_g ? exp(lom.a₀ + lom.a₁ * log(K[t])) : exp(lom.b₀ + lom.b₁ * log(K[t])))
        end
        res.K_path = K
        a₀, a₁, b₀, b₁, R² = KSVFI.estimate_regression(prim, res, sim)
        @test [a₀, a₁, b₀, b₁] ≈ collect(lom) rtol=1e-8
        @test R² ≈ 1
    end

    @testset "equilibrium law of motion" begin
        prim, sho, sim, res = quietly(() -> KSVFI.solve_model(; prim=prim_coarse,
            sim=KSVFI.Simulations(; N=1000, T=1500, burn=300, V_tol=1e-7, reg_tol=1e-4, reg_max_iter=40)))
        # Converged: re-estimating on the final simulation reproduces the law of motion households used
        a₀, a₁, b₀, b₁, R² = KSVFI.estimate_regression(prim, res, sim)
        @test maxdiff([a₀, a₁, b₀, b₁], [res.a₀, res.a₁, res.b₀, res.b₁]) < sim.reg_tol
        # Approximate aggregation (Krusell and Smith 1998): log K' is nearly linear in log K
        @test R² > 0.99
        # The law of motion is stable in both states and accumulates more capital in good times
        @test 0 < res.a₁ < 1 && 0 < res.b₁ < 1
        K_good, K_bad = exp(res.a₀ / (1 - res.a₁)), exp(res.b₀ / (1 - res.b₁))
        K_sim = res.K_path[sim.burn:end]
        @test K_bad < mean(K_sim) < K_good
        @test prim.K_min <= minimum(K_sim) && maximum(K_sim) <= prim.K_max  # no extrapolation off the K grid
    end
end
