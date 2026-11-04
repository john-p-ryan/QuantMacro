# Shared setup and economic helpers for the aggregate uncertainty test suite.

# Silences the solvers' progress messages.
quietly(f) = redirect_stdout(f, devnull)

maxdiff(a, b) = maximum(abs.(a .- b))
nanmax(x) = maximum(filter(!isnan, x))

# Size of the TFP shock (1%) and the IRF horizon for the linear methods, as in methods_comparison.jl.
const SHOCK = 0.01
const HORIZON = 200


# --- Prices ---

# Prices from the Cobb-Douglas firm's first-order conditions at capital K and TFP Z.
rental_rate(prim, K, Z=1.0) = prim.α * Z * (prim.L / K)^(1 - prim.α)
wage(prim, K, Z=1.0) = (1 - prim.α) * Z * (K / prim.L)^prim.α
capital_demand(prim, r) = prim.L * (prim.α / r)^(1 / (1 - prim.α))

# Long-run distribution of a Markov chain: the left unit eigenvector of P.
function stationary(P)
    π = nullspace(Matrix(P' - I))[:, 1]
    return π ./ sum(π)
end


# --- Solutions shared across test files, each computed once on first use ---

const CACHE = Dict{Symbol, Any}()
cached(f, key) = get!(f, CACHE, key)

# Aiyagari steady state on the default grids, around which BKM, Reiter, and SSJ linearize.
steady_state() = cached(:steady_state) do
    quietly(() -> Linear.solve_model())
end

# Relative capital-market clearing error of the steady state. The linear methods anchor capital at res.K, the
# household supply, and prices at res.r, so their IRFs inherit this error divided by the shock size.
function market_gap()
    prim, res = steady_state()
    return abs(res.K / capital_demand(prim, res.r) - 1)
end

# BKM transition after a 1% MIT shock to TFP, and its IRFs (percent deviations per percent shock).
bkm() = cached(:bkm) do
    prim, res = steady_state()
    sim, tr = quietly(() -> Linear.SolveTransition_from_SS(prim, res; T=HORIZON, Z_shock=SHOCK))
    (sim=sim, tr=tr, irfs=Linear.ComputeIRFs(prim, res, tr, sim))
end

# Reiter's linearized system, its QZ solution, and the IRFs to a 1% shock (in percent, so per percent shock).
reiter() = cached(:reiter) do
    prim, res = steady_state()
    system = quietly(() -> Linear.construct_reiter_system(prim, res))
    g_x, h_x, eu = Linear.solve_eig(hcat(system.J_x′, system.J_y′), -hcat(system.J_x, system.J_y), system.n_x)
    irfs = Linear.calculate_reiter_irfs(prim, res, system, g_x, h_x; shock_size=SHOCK, T=HORIZON)
    (system=system, g_x=g_x, h_x=h_x, eu=eu, irfs=irfs)
end

# SSJ household and firm Jacobians, and the general equilibrium response of capital to a 1% shock.
ssj() = cached(:ssj) do
    prim, res = steady_state()
    J_K_r, J_K_w = quietly(() -> Linear.fast_jacobian(prim, res, 1e-5, 1e-5, HORIZON))
    J_r_K, J_w_K, J_r_Z, J_w_Z = Linear.Firm_Block(prim, res, HORIZON)
    dZ, dK = Linear.solve_dK(0.75, SHOCK, J_K_r, J_K_w, J_r_K, J_w_K, J_r_Z, J_w_Z)
    (J_K_r=J_K_r, J_K_w=J_K_w, J_r_K=J_r_K, J_w_K=J_w_K, J_r_Z=J_r_Z, J_w_Z=J_w_Z,
     dZ=dZ, dK=dK, irf_K=dK ./ (res.K * SHOCK))
end

# Aiyagari steady state inside KS_EGM_hist.jl, which initializes its distribution.
ks_aiyagari() = cached(:ks_aiyagari) do
    quietly(() -> KSEGM.Aiyagari.solve_model())
end

# KS (EGM and histogram) solution on a coarse aggregate capital grid with a short simulation.
ks_egm() = cached(:ks_egm) do
    prim, sim, res = quietly(() -> KSEGM.SolveModel(; nK=7, T=3000, burn=500, reg_tol=1e-6))
    (prim=prim, sim=sim, res=res)
end


# --- Krusell-Smith helpers ---

# Tomorrow's aggregate capital from a solution's law of motion, clamped to the aggregate capital grid.
function ks_K_next(prim, res, K, z)
    K_next = z == prim.z_g ? exp(res.a₀ + res.a₁ * log(K)) : exp(res.b₀ + res.b₁ * log(K))
    return clamp(K_next, prim.K_min, prim.K_max)
end

# Cash on hand (1 + r - δ)k + w ē ϵ for every (k, ϵ, K, z) on the policy grids of KS_EGM_hist.jl.
ks_cash_on_hand(prim) = [(1 + rental_rate(prim, K, z) - prim.δ) * k + wage(prim, K, z) * prim.ē * ϵ
                         for k in prim.k_grid, ϵ in prim.ϵ_grid, K in prim.K_grid, z in prim.z_grid]

# Relative Euler-equation errors |ĉ / c - 1| of a KS_EGM_hist.jl solution, where
# ĉ = u'⁻¹(E[β(1 + r' - δ) u'(c(k', K', ϵ', z'))]). The consumption policy is interpolated by PCHIP in k, since k' is
# off the grid and linear interpolation of the concave c(k) would dominate the error, and linearly in K, as in the
# solver. NaN where the borrowing constraint binds and the Euler equation need not hold.
function ks_euler_errors(prim, res; binding_tol=1e-6)
    (; β, γ, δ, k_grid, K_grid, ϵ_grid, z_grid, Mzz, M_unemp, k_min) = prim
    c_splines = [PchipSpline(k_grid, res.c_policy[:, iϵ, iK, iz])
                 for iϵ in eachindex(ϵ_grid), iK in eachindex(K_grid), iz in eachindex(z_grid)]
    errors = fill(NaN, size(res.c_policy))
    for (iz, z) in enumerate(z_grid), (iK, K) in enumerate(K_grid)
        K′ = ks_K_next(prim, res, K, z)
        j = clamp(searchsortedlast(K_grid, K′), 1, length(K_grid) - 1)  # K' lies in [K_grid[j], K_grid[j+1]]
        θ = (K′ - K_grid[j]) / (K_grid[j+1] - K_grid[j])
        for iϵ in eachindex(ϵ_grid)
            k′ = res.k_policy[:, iϵ, iK, iz]
            Emu = zeros(length(k_grid))
            for iz′ in eachindex(z_grid), iϵ′ in eachindex(ϵ_grid)
                R′ = 1 + rental_rate(prim, K′, z_grid[iz′]) - δ
                c′ = (1 - θ) .* evaluate_spline(c_splines[iϵ′, j, iz′], k′) .+ θ .* evaluate_spline(c_splines[iϵ′, j+1, iz′], k′)
                Emu .+= Mzz[iz, iz′] * M_unemp[iϵ, iϵ′] * β * R′ .* c′ .^ (-γ)
            end
            free = k′ .> k_min + binding_tol
            errors[free, iϵ, iK, iz] = abs.(Emu[free] .^ (-1 / γ) ./ res.c_policy[free, iϵ, iK, iz] .- 1)
        end
    end
    return errors
end

# Generalized IRF of capital in a KS solution, in percent deviations per percent TFP shock, with period 1 (the shock
# period, when capital is predetermined) first so that it lines up with the linear methods' IRFs.
function ks_irf(prim, res, sim; horizon=HORIZON, num_simulations=100_000, seed=1234)
    Random.seed!(seed)
    irf_K, _ = KSEGM.estimate_impulse_response(prim, res, sim; num_simulations, max_horizon=horizon)
    K̄ = mean(res.K_sim_path[(sim.burn+1):end])
    shock = prim.z_g - (prim.z_g + prim.z_b) / 2  # z_g minus unconditional mean TFP (symmetric chain)
    return [0.0; irf_K[1:horizon-1] ./ (K̄ * shock)]
end
