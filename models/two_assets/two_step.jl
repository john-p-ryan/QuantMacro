using LinearAlgebra
using Spline

# ============================================================
# Parameters
# ============================================================
struct Params
    β::Float64      # discount factor
    σ::Float64      # CRRA coefficient
    R_k::Float64    # gross return on illiquid asset
    R_b::Float64    # gross return on liquid asset
    b_min::Float64  # borrowing constraint on liquid asset
    k_min::Float64  # lower bound on illiquid asset
    χ::Float64      # adjustment cost scale parameter
end

function Params(;
    β = 0.96,
    σ = 2.0,
    R_k = 1.04,
    R_b = 1.01,
    b_min = 0.0,
    k_min = 0.0,
    χ = 0.05
)
    Params(β, σ, R_k, R_b, b_min, k_min, χ)
end

# ============================================================
# Grids and income process
# ============================================================
struct Grids
    k_grid::Vector{Float64}
    b_grid::Vector{Float64}       # regular grid for b (b ≥ b_min)
    b_wide_grid::Vector{Float64}  # extended grid for V_NA (extends below b_min)
    z_vals::Vector{Float64}
    Π::Matrix{Float64}            # transition matrix for z
end

function make_grids(p::Params;
    n_k = 30, n_b = 50, n_z = 2,
    k_max = 10.0, b_max = 10.0,
    n_b_wide = 80, b_wide_min = nothing
)
    k_grid = range(p.k_min, k_max, length = n_k) |> collect
    b_grid = range(p.b_min, b_max, length = n_b) |> collect

    # Auto-compute wide grid lower bound: cover plausible b* values
    # b* = b + (R_k/R_b)(k - k') - g(k,k')/R_b  can go below b_min when k' > k
    if b_wide_min === nothing
        b_wide_min = p.b_min - 2.0 * (b_max - p.b_min)
    end
    b_wide_grid = range(b_wide_min, b_max, length = n_b_wide) |> collect

    # Simple 2-state Markov for income (matches grid_search.jl)
    z_vals = [0.25, 1.0]
    Π = [0.8  0.2;
         0.2  0.8]

    Grids(k_grid, b_grid, b_wide_grid, z_vals, Π)
end

# ============================================================
# Utility and adjustment cost
# ============================================================
@inline function u(c, σ)
    c <= 0.0 && return -1e10
    σ == 1.0 ? log(c) : (c^(1 - σ) - 1) / (1 - σ)
end

@inline function u_prime(c, σ)
    c <= 0.0 && return 1e10
    c^(-σ)
end

@inline function u_prime_inv(x, σ)
    x <= 0.0 && return 1e10
    x^(-1.0 / σ)
end

@inline function adj_cost(k, k′, χ)
    χ * (k - k′)^2
end

# ============================================================
# Two-step Bellman update (Graves' method)
#
# 1. Solve the No-Adjust problem on a wide b grid using EGM.
# 2. Recover V_tilde(k, b, z; k') = V_NA(k', b*, z) via the
#    mapping b* = b + (R_k/R_b)(k - k') - g(k, k')/R_b.
# 3. Maximize over k' to get the updated V and policies.
# ============================================================
function bellman_two_step!(V_new, pol_k, pol_b, pol_c,
                           V, c_pol, p::Params, g::Grids)

    n_k  = length(g.k_grid)
    n_b  = length(g.b_grid)
    n_bw = length(g.b_wide_grid)
    n_z  = length(g.z_vals)

    # Intermediate storage for the no-adjust problem
    V_NA     = Array{Float64}(undef, n_k, n_bw, n_z)
    c_NA     = Array{Float64}(undef, n_k, n_bw, n_z)
    b_pol_NA = Array{Float64}(undef, n_k, n_bw, n_z)

    # ──────────────────────────────────────────────────────────
    # Step 1: Solve the No-Adjust problem via EGM
    #
    #   V_NA(k, b, z) = max_{c, b'} u(c) + β E[V(k, b', z')]
    #   s.t.  c + b' = (R_k - 1)k + R_b b + z,  b' ≥ b_min
    #
    # The continuation uses the full V (agents can adjust next
    # period), and the problem is solved on the wide b grid.
    # ──────────────────────────────────────────────────────────

    # Precompute V splines in the b dimension for continuation
    V_b_spl = Matrix{PchipSplineInterpolation}(undef, n_k, n_z)
    for iz in 1:n_z, ik in 1:n_k
        V_b_spl[ik, iz] = PchipSpline(g.b_grid, V[ik, :, iz])
    end

    for i_z in 1:n_z
        z = g.z_vals[i_z]

        for i_k in 1:n_k
            k = g.k_grid[i_k]
            income = (p.R_k - 1) * k + z   # cash flow excluding liquid wealth

            # -- 1a. Expected marginal utility at each b' on the regular grid --
            EMU = Vector{Float64}(undef, n_b)
            @inbounds for i_bp in 1:n_b
                s = 0.0
                for iz′ in 1:n_z
                    s += g.Π[i_z, iz′] * u_prime(c_pol[i_k, i_bp, iz′], p.σ)
                end
                EMU[i_bp] = s
            end

            # -- 1b. EGM: invert Euler for the unconstrained region --
            #   c = (u')⁻¹(β R_b EMU),  then b = (c + b' - income) / R_b
            c_endo = Vector{Float64}(undef, n_b)
            b_endo = Vector{Float64}(undef, n_b)
            @inbounds for j in 1:n_b
                c_endo[j] = u_prime_inv(p.β * p.R_b * EMU[j], p.σ)
                b_endo[j] = (c_endo[j] + g.b_grid[j] - income) / p.R_b
            end

            # -- 1c. Interpolate onto the wide grid --
            # Safety: enforce strict monotonicity of b_endo (required by PchipSpline).
            # This is guaranteed when c_pol is monotone in b, but may fail in
            # early iterations with a poor initial guess.
            for j in 2:n_b
                if b_endo[j] <= b_endo[j-1]
                    b_endo[j] = b_endo[j-1] + 1e-12
                end
            end
            c_spl  = PchipSpline(b_endo, c_endo)
            bp_spl = PchipSpline(b_endo, g.b_grid)

            # Constrained region: b ≤ b_endo[1]  →  b' = b_min
            i_first_unc = n_bw + 1  # default: all constrained
            @inbounds for i_bw in 1:n_bw
                bw = g.b_wide_grid[i_bw]
                if bw > b_endo[1]
                    i_first_unc = i_bw
                    break
                end
                b_pol_NA[i_k, i_bw, i_z] = p.b_min
                c_NA[i_k, i_bw, i_z]     = income + p.R_b * bw - p.b_min
            end

            # Unconstrained region: batch-evaluate splines
            if i_first_unc <= n_bw
                bw_unc = g.b_wide_grid[i_first_unc:end]
                c_NA[i_k, i_first_unc:end, i_z] = evaluate_spline(c_spl, bw_unc)
                bp_raw = evaluate_spline(bp_spl, bw_unc)
                @inbounds for (j, idx) in enumerate(i_first_unc:n_bw)
                    b_pol_NA[i_k, idx, i_z] = clamp(bp_raw[j], p.b_min, g.b_grid[end])
                end
            end

            # -- 1d. Compute V_NA = u(c) + β E[V(k, b', z')] --
            bp_vals = @view b_pol_NA[i_k, :, i_z]
            bp_vec  = collect(bp_vals)   # spline needs a plain Vector
            EV = zeros(n_bw)
            for iz′ in 1:n_z
                EV .+= g.Π[i_z, iz′] .* evaluate_spline(V_b_spl[i_k, iz′], bp_vec)
            end
            @inbounds for i_bw in 1:n_bw
                V_NA[i_k, i_bw, i_z] = u(c_NA[i_k, i_bw, i_z], p.σ) + p.β * EV[i_bw]
            end
        end
    end

    # ──────────────────────────────────────────────────────────
    # Step 2: Build V_tilde via  V_tilde(k,b,z;k') = V_NA(k', b*, z)
    # Step 3: Maximize over k' to get updated V and policies
    # ──────────────────────────────────────────────────────────

    # Splines of V_NA and b_pol_NA in the b_wide dimension
    V_NA_spl  = Matrix{PchipSplineInterpolation}(undef, n_k, n_z)
    bp_NA_spl = Matrix{PchipSplineInterpolation}(undef, n_k, n_z)
    for iz in 1:n_z, ik in 1:n_k
        V_NA_spl[ik, iz]  = PchipSpline(g.b_wide_grid, V_NA[ik, :, iz])
        bp_NA_spl[ik, iz] = PchipSpline(g.b_wide_grid, b_pol_NA[ik, :, iz])
    end

    # For each candidate k', batch-evaluate V_NA at all (k, b) pairs
    b_star_vec = Vector{Float64}(undef, n_k * n_b)
    V_tilde    = Array{Float64}(undef, n_k, n_b, n_k)

    for i_z in 1:n_z

        # -- Evaluate V_tilde for every (k, b, k') combination --
        for ik′ in 1:n_k
            k′ = g.k_grid[ik′]
            idx = 0
            @inbounds for i_b in 1:n_b
                b = g.b_grid[i_b]
                for i_k in 1:n_k
                    k = g.k_grid[i_k]
                    idx += 1
                    b_star_vec[idx] = b + (p.R_k / p.R_b) * (k - k′) -
                                      adj_cost(k, k′, p.χ) / p.R_b
                end
            end
            vals = evaluate_spline(V_NA_spl[ik′, i_z], b_star_vec)
            # Reshape: vals is ordered (i_k fast, i_b slow)
            # Mark infeasible choices (resources < b_min ⟹ c < 0) as -Inf
            @inbounds for i_b in 1:n_b, i_k in 1:n_k
                resources = p.R_k * g.k_grid[i_k] + p.R_b * g.b_grid[i_b] +
                            g.z_vals[i_z] - k′ - adj_cost(g.k_grid[i_k], k′, p.χ)
                if resources <= p.b_min
                    V_tilde[i_k, i_b, ik′] = -Inf
                else
                    V_tilde[i_k, i_b, ik′] = vals[(i_b - 1) * n_k + i_k]
                end
            end
        end

        # -- Maximize over k' --
        @inbounds for i_b in 1:n_b, i_k in 1:n_k
            best_val = V_tilde[i_k, i_b, 1]
            best_ik  = 1
            for ik′ in 2:n_k
                v = V_tilde[i_k, i_b, ik′]
                if v > best_val
                    best_val = v
                    best_ik  = ik′
                end
            end
            V_new[i_k, i_b, i_z] = best_val
            pol_k[i_k, i_b, i_z] = best_ik
        end

        # -- Extract consumption and b' policies --
        # Group (i_k, i_b) pairs by optimal k' index to batch spline evaluations
        for ik′ in 1:n_k
            indices = findall(@view(pol_k[:, :, i_z]) .== ik′)
            isempty(indices) && continue

            k′ = g.k_grid[ik′]
            n_pts = length(indices)
            bs_vec = Vector{Float64}(undef, n_pts)
            @inbounds for (j, ci) in enumerate(indices)
                i_k, i_b = ci.I
                bs_vec[j] = g.b_grid[i_b] +
                            (p.R_k / p.R_b) * (g.k_grid[i_k] - k′) -
                            adj_cost(g.k_grid[i_k], k′, p.χ) / p.R_b
            end

            bp_vals = evaluate_spline(bp_NA_spl[ik′, i_z], bs_vec)

            @inbounds for (j, ci) in enumerate(indices)
                i_k, i_b = ci.I
                resources = p.R_k * g.k_grid[i_k] + p.R_b * g.b_grid[i_b] +
                            g.z_vals[i_z] - k′ - adj_cost(g.k_grid[i_k], k′, p.χ)
                # Clamp b': respect borrowing constraint and ensure c > 0
                bp_max = max(p.b_min, min(g.b_grid[end], resources - 1e-10))
                bp = clamp(bp_vals[j], p.b_min, bp_max)
                pol_b[i_k, i_b, i_z] = bp
                pol_c[i_k, i_b, i_z] = resources - bp
            end
        end
    end
end

# ============================================================
# Value Function Iteration
# ============================================================
function solve_vfi(p::Params, g::Grids; tol = 1e-6, max_iter = 1000, verbose = true)
    n_k = length(g.k_grid)
    n_b = length(g.b_grid)
    n_z = length(g.z_vals)

    V     = zeros(n_k, n_b, n_z)
    V_new = similar(V)
    pol_k = ones(Int, n_k, n_b, n_z)
    pol_b = zeros(n_k, n_b, n_z)
    pol_c = zeros(n_k, n_b, n_z)

    # Initial consumption policy: consume all resources, save b_min
    c_pol = zeros(n_k, n_b, n_z)
    for i_z in 1:n_z, i_b in 1:n_b, i_k in 1:n_k
        c_pol[i_k, i_b, i_z] = p.R_k * g.k_grid[i_k] + p.R_b * g.b_grid[i_b] +
                                 g.z_vals[i_z] - g.k_grid[i_k] - p.b_min
    end

    for iter in 1:max_iter
        bellman_two_step!(V_new, pol_k, pol_b, pol_c, V, c_pol, p, g)
        err = maximum(abs.(V_new .- V))

        if verbose && iter % 10 == 0
            println("Iter $iter, ||V_new - V|| = $err")
        end

        if err < tol
            verbose && println("Converged in $iter iterations (err = $err)")
            copy!(V, V_new)
            c_pol .= pol_c
            break
        end

        copy!(V, V_new)
        c_pol .= pol_c

        if iter == max_iter
            @warn "VFI did not converge after $max_iter iterations (err = $err)"
        end
    end

    # Convert index policies to levels
    k_pol = zeros(n_k, n_b, n_z)
    for i in eachindex(pol_k)
        k_pol[i] = g.k_grid[pol_k[i]]
    end

    (; V, k_pol, b_pol = pol_b, c_pol, pol_k_idx = pol_k)
end

# ============================================================
# Run
# ============================================================
p = Params()
g = make_grids(p; n_k = 250, n_b = 90, n_z = 2, k_max = 25.0, b_max = 9.0,
               n_b_wide = 120, b_wide_min = -10.0)

println("Solving two-asset model via Graves' two-step method...")
@time sol = solve_vfi(p, g; tol = 1e-6, max_iter = 500)
println("Done.")

using Plots

# k' policy
plot(g.k_grid, sol.k_pol[:, 1, :],  label=["b low, z low" "b low, z high"])
plot!(g.k_grid, sol.k_pol[:, 45, :], label=["b mid, z low" "b mid, z high"])
plot!(g.k_grid, sol.k_pol[:, 90, :], label=["b high, z low" "b high, z high"],
      xlabel="k", ylabel="k'", title="Illiquid Asset Policy (Two-Step)")

# b' policy
plot(g.k_grid, sol.b_pol[:, 1, :],  label=["b low, z low" "b low, z high"])
plot!(g.k_grid, sol.b_pol[:, 45, :], label=["b mid, z low" "b mid, z high"])
plot!(g.k_grid, sol.b_pol[:, 90, :], label=["b high, z low" "b high, z high"],
      xlabel="k", ylabel="b'", title="Liquid Asset Policy (Two-Step)")

# Value function
plot(g.k_grid, sol.V[:, 1, :],  label=["b low, z low" "b low, z high"])
plot!(g.k_grid, sol.V[:, 45, :], label=["b mid, z low" "b mid, z high"])
plot!(g.k_grid, sol.V[:, 90, :], label=["b high, z low" "b high, z high"],
      xlabel="k", ylabel="V", title="Value Function (Two-Step)")
