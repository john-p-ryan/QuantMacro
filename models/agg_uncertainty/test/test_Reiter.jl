@testset "Reiter (Reiter.jl)" begin
    prim, res = steady_state()

    @testset "QZ solution of known linear models" begin
        # A state x' = ρx and a forward-looking control y = x + φy', so y = x / (1 - φρ). The system is A [x'; y'] = B [x; y].
        ρ, φ = 0.9, 0.5
        A = [1.0 0.0; 0.0 -φ]
        B = [ρ 0.0; 1.0 -1.0]
        g_x, h_x, eu = Linear.solve_eig(A, B, 1)
        @test eu == [true, true]
        @test h_x ≈ [ρ;;] && g_x ≈ [1 / (1 - φ * ρ);;]
        # With φ > 1 both roots are stable, the equilibrium is indeterminate, and no solution is returned
        @test quietly(() -> Linear.solve_eig([1.0 0.0; 0.0 -2.0], B, 1)) === nothing

        # Two states rotating and decaying, x' = Hx, whose complex pair of eigenvalues must be selected together, and a
        # control y = cx + φy', so y = c (I - φH)⁻¹ x
        θ = 0.6
        H = 0.9 * [cos(θ) -sin(θ); sin(θ) cos(θ)]
        c = [1.0 2.0]
        A = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 -φ]
        B = [H zeros(2); c -1.0]
        g_x, h_x, eu = Linear.solve_eig(A, B, 2)
        @test eu == [true, true]
        @test h_x ≈ H && g_x ≈ c / (I - φ * H)
    end

    (; system, g_x, h_x, eu, irfs) = reiter()
    n_μ = system.n_x - 1  # distribution states; the first state is TFP
    tfp = n_μ + 1         # the TFP equation follows the distribution equations

    @testset "steady state of the discretized system" begin
        f_ss = system.f(system.v_ss)
        @test quietly(() -> Linear.verify_steady_state(system)) == maximum(abs.(f_ss))
        @test maximum(abs.(f_ss[1:n_μ])) < 1e-10  # the histogram is stationary
        @test f_ss[tfp] == 0
        # Euler residuals relative to u'(c). The steady state imposes the Euler equation at EGM's endogenous grid points,
        # so on the fixed grid it holds up to PCHIP's interpolation error, which is largest where the constraint
        # starts to bind (an unemployed household with k ≈ 0.02 and c ≈ 0.009).
        euler = reshape(f_ss[tfp+1:end], prim.nk, prim.nz) ./ Linear.u_prime.(res.c_policy, prim.γ)
        @test maximum(abs.(euler)) < 5e-3
        @test maximum(abs.(euler[prim.k_grid .>= 1, :])) < 1e-4
    end

    @testset "linearized system" begin
        # As many equations as unknowns: the TFP state, the histogram less one mass per employment state, the policy
        @test system.n_x == 1 + prim.nz * (prim.n_hist - 1) && system.n_y == prim.nk * prim.nz
        @test size(system.J_x) == (system.n_x + system.n_y, system.n_x)
        # Only the TFP equation log Z' = ρ log Z + σε involves the innovation
        @test findall(!iszero, vec(system.J_ε)) == [tfp] && system.J_ε[tfp] ≈ -system.σ
        @test system.J_x[tfp, 1] ≈ -system.ρ && system.J_x′[tfp, 1] ≈ 1
        # ForwardDiff's Jacobian agrees with central differences of f in TFP and in an interior policy entry
        h = 1e-6
        J = hcat(system.J_x, system.J_y, system.J_x′, system.J_y′)
        for j in (1, system.n_x + prim.nk ÷ 2)
            e = zeros(length(system.v_ss))
            e[j] = h
            @test (system.f(system.v_ss + e) - system.f(system.v_ss - e)) / 2h ≈ J[:, j] rtol=1e-5
        end
    end

    @testset "solution" begin
        # Blanchard-Kahn conditions hold and the state transition is stable
        @test eu == [true, true]
        @test maximum(abs.(eigvals(h_x))) < 1
        # TFP is exogenous: its row of h_x is ρ on itself and zero on the distribution
        @test h_x[1, 1] ≈ system.ρ && maximum(abs.(h_x[1, 2:end])) < 1e-12
    end

    @testset "impulse responses" begin
        @test irfs.Z ≈ 100 * SHOCK .* system.ρ .^ (0:HORIZON)
        K = irfs.K
        peak = argmax(K)
        @test abs(K[1]) < 1e-10  # the distribution, and so capital, is predetermined on impact
        @test all(K[2:100] .> 0) && 5 < peak < 30
        @test irfs.r[1] ≈ 100 * SHOCK atol=0.01
        @test irfs.w[1] ≈ 100 * SHOCK atol=0.01
        @test irfs.r[peak] < 0
        # The solution is linear in the shock
        twice = Linear.calculate_reiter_irfs(prim, res, system, g_x, h_x; shock_size=2SHOCK, T=HORIZON)
        @test twice.K ≈ 2 .* K
    end

    @testset "simulation" begin
        # Simulating along the MIT shock's TFP path reproduces the impulse response
        Z_path = exp.(SHOCK .* system.ρ .^ (0:HORIZON))
        path = Linear.simulate_reiter_economy(Z_path, system, g_x, h_x, res, prim)
        @test 100 .* (path.K ./ res.K .- 1) ≈ irfs.K atol=1e-8
        # Without shocks the economy stays at the steady state
        flat = Linear.simulate_reiter_economy(ones(20), system, g_x, h_x, res, prim)
        @test flat.K ≈ fill(res.K, 20) && flat.K_var ≈ fill(res.K_var, 20)
    end
end
