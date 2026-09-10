using LinearAlgebra
using Spline

# ============================================================
# Parameters
# ============================================================
struct Params
    β::Float64
    σ::Float64
    R_k::Float64
    R_b::Float64
    b_min::Float64
    k_min::Float64
    χ::Float64
end

function Params(;
    β  = 0.96,
    σ  = 2.0,
    R_k = 1.04,
    R_b = 1.01,
    b_min = 0.0,
    k_min = 0.0,
    χ  = 0.05
)
    Params(β, σ, R_k, R_b, b_min, k_min, χ)
end

# ============================================================
# Grids and income process
# ============================================================
struct Grids
    k_grid::Vector{Float64}
    b_grid::Vector{Float64}
    z_vals::Vector{Float64}
    Π::Matrix{Float64}
end

function make_grids(p::Params; n_k=30, n_b=50, k_max=10.0, b_max=10.0)
    k_grid = collect(range(p.k_min, k_max, length=n_k))
    b_grid = collect(range(p.b_min, b_max, length=n_b))
    z_vals = [0.25, 1.0]
    Π = [0.8  0.2;
         0.2  0.8]
    Grids(k_grid, b_grid, z_vals, Π)
end

# ============================================================
# Utility and adjustment cost
# ============================================================
const _EPS     = 1e-10
const _PENALTY = -1e10

@inline function u(c::Float64, σ::Float64)
    c <= 0.0 && return _PENALTY
    σ == 1.0 ? log(c) : (c^(1.0 - σ) - 1.0) / (1.0 - σ)
end

@inline function u_prime(c::Float64, σ::Float64)
    max(c, _EPS)^(-σ)
end

@inline function u_prime_inv(x::Float64, σ::Float64)
    max(x, _EPS)^(-1.0 / σ)
end

@inline function adj_cost(k::Float64, k′::Float64, χ::Float64)
    χ * (k - k′)^2
end

# Clamp to spline domain before evaluating (mimics Python's _safe_eval /
# PchipInterpolator(extrapolate=False) + clip pattern).
function safe_eval(spl::PchipSplineInterpolation, x::Vector{Float64})
    evaluate_spline(spl, clamp.(x, spl.x[1], spl.x[end]))
end

# ============================================================
# Step 1: No-Adjust sub-problem via EGM
#
# V_NA(k, b, z) = max_{c,b'} u(c) + β E[V(k, b', z')]
# s.t.  c + b' = (R_k-1)*k + R_b*b + z,  b' >= b_min
#
# EGM: treat b' as exogenous on b_grid, invert Euler for c and b today.
# Returns splines indexed by [ik, iz]:
#   V_NA_spl[ik, iz](b)  →  V_NA(k_grid[ik], b, z_vals[iz])
#   bp_NA_spl[ik, iz](b) →  optimal b' at (k_grid[ik], b, z_vals[iz])
# Domain covers b_floor to b_endo[end], accommodating all b* queries.
# ============================================================
function egm_no_adjust(V, c_pol, p::Params, g::Grids, b_floor::Float64)
    n_k  = length(g.k_grid)
    n_b  = length(g.b_grid)
    n_z  = length(g.z_vals)
    N_CON = 30  # extra points in constrained region

    V_NA_spl  = Matrix{PchipSplineInterpolation}(undef, n_k, n_z)
    bp_NA_spl = Matrix{PchipSplineInterpolation}(undef, n_k, n_z)

    for ik in 1:n_k
        k = g.k_grid[ik]
        income_k = (p.R_k - 1.0) * k   # interest income on illiquid asset

        for iz in 1:n_z
            z = g.z_vals[iz]

            # -- EGM: b' on b_grid, invert Euler --
            # EMU[j] = β R_b Σ_z' Π[iz,z'] u'(c(k, b_grid[j], z'))
            EMU = Vector{Float64}(undef, n_b)
            @inbounds for j in 1:n_b
                s = 0.0
                for iz2 in 1:n_z
                    s += g.Π[iz, iz2] * u_prime(c_pol[ik, j, iz2], p.σ)
                end
                EMU[j] = p.β * p.R_b * s
            end

            c_endo = [u_prime_inv(EMU[j], p.σ) for j in 1:n_b]
            b_endo = [(c_endo[j] + g.b_grid[j] - income_k - z) / p.R_b for j in 1:n_b]

            # EV[j] = β Σ_z' Π[iz,z'] V(k, b_grid[j], z') — no spline needed
            EV = Vector{Float64}(undef, n_b)
            @inbounds for j in 1:n_b
                s = 0.0
                for iz2 in 1:n_z
                    s += g.Π[iz, iz2] * V[ik, j, iz2]
                end
                EV[j] = p.β * s
            end

            V_NA_endo = [u(c_endo[j], p.σ) + EV[j] for j in 1:n_b]

            # -- Constrained region: b' = b_min for b < b_endo[1] --
            b_feas_min = (p.b_min - income_k - z) / p.R_b + 1e-8
            b_con_lo   = max(b_floor, b_feas_min)
            EV_bmin    = p.β * dot(g.Π[iz, :], V[ik, 1, :])

            if b_con_lo < b_endo[1] - 1e-10
                b_con    = collect(range(b_con_lo, b_endo[1] - 1e-10, length=N_CON))
                c_con    = [income_k + p.R_b * b_con[j] + z - p.b_min for j in 1:N_CON]
                V_NA_con = [u(c_con[j], p.σ) + EV_bmin for j in 1:N_CON]
                bp_con   = fill(p.b_min, N_CON)
            else
                b_con    = Float64[]
                V_NA_con = Float64[]
                bp_con   = Float64[]
            end

            # -- Assemble, sort, deduplicate, build PCHIP splines --
            b_all  = vcat(b_con, b_endo)
            V_all  = vcat(V_NA_con, V_NA_endo)
            bp_all = vcat(bp_con, g.b_grid)

            order  = sortperm(b_all)
            b_all  = b_all[order]
            V_all  = V_all[order]
            bp_all = bp_all[order]

            # Remove duplicates (PCHIP requires strictly increasing x)
            keep = vcat(true, diff(b_all) .> 1e-12)
            b_all  = b_all[keep]
            V_all  = V_all[keep]
            bp_all = bp_all[keep]

            V_NA_spl[ik, iz]  = PchipSpline(b_all, V_all;  extrapolate=false)
            bp_NA_spl[ik, iz] = PchipSpline(b_all, bp_all; extrapolate=false)
        end
    end

    return V_NA_spl, bp_NA_spl
end


# ============================================================
# Step 2 & 3: Maximise over k'
#
# V_tilde(k, b, z; k') = V_NA(k', b*, z)
# where b* = b + (R_k/R_b)(k - k') - g(k,k')/R_b
#
# For each state (k, b, z), maximise over k' on the grid.
# ============================================================
function maximise_over_kprime(V_NA_spl, bp_NA_spl, p::Params, g::Grids)
    n_k = length(g.k_grid)
    n_b = length(g.b_grid)
    n_z = length(g.z_vals)

    V_new  = fill(-Inf, n_k, n_b, n_z)
    kp_idx = ones(Int, n_k, n_b, n_z)

    b_star_vec  = Vector{Float64}(undef, n_k * n_b)
    feasible_vec = Vector{Bool}(undef, n_k * n_b)

    for iz in 1:n_z
        z = g.z_vals[iz]

        for ikp in 1:n_k
            kp     = g.k_grid[ikp]
            spl_V  = V_NA_spl[ikp, iz]
            b_lo   = spl_V.x[1]
            b_hi   = spl_V.x[end]

            # Build b* for all (ik, ib) pairs (flat, column-major: ik varies fastest)
            @inbounds for ib in 1:n_b
                b = g.b_grid[ib]
                for ik in 1:n_k
                    k   = g.k_grid[ik]
                    gc  = adj_cost(k, kp, p.χ)
                    res = p.R_k * k + p.R_b * b + z - kp - gc
                    bs  = b + (p.R_k / p.R_b) * (k - kp) - gc / p.R_b
                    flat = (ib - 1) * n_k + ik
                    b_star_vec[flat]   = clamp(bs, b_lo, b_hi)
                    feasible_vec[flat] = res > p.b_min + _EPS
                end
            end

            vals_vec = evaluate_spline(spl_V, b_star_vec)

            @inbounds for ib in 1:n_b, ik in 1:n_k
                flat = (ib - 1) * n_k + ik
                v    = feasible_vec[flat] ? vals_vec[flat] : _PENALTY
                if v > V_new[ik, ib, iz]
                    V_new[ik, ib, iz]  = v
                    kp_idx[ik, ib, iz] = ikp
                end
            end
        end
    end

    # -- Recover k', b', c policies --
    k_pol = zeros(n_k, n_b, n_z)
    b_pol = zeros(n_k, n_b, n_z)
    c_pol = zeros(n_k, n_b, n_z)

    for iz in 1:n_z
        z = g.z_vals[iz]
        for ikp in 1:n_k
            kp     = g.k_grid[ikp]
            spl_bp = bp_NA_spl[ikp, iz]

            indices = findall(kp_idx[:, :, iz] .== ikp)
            isempty(indices) && continue

            n_pts  = length(indices)
            bs_vec = Vector{Float64}(undef, n_pts)
            @inbounds for (j, ci) in enumerate(indices)
                ik, ib = ci.I
                k  = g.k_grid[ik];  b = g.b_grid[ib]
                gc = adj_cost(k, kp, p.χ)
                bs = b + (p.R_k / p.R_b) * (k - kp) - gc / p.R_b
                bs_vec[j] = clamp(bs, spl_bp.x[1], spl_bp.x[end])
            end

            bp_vals = evaluate_spline(spl_bp, bs_vec)

            @inbounds for (j, ci) in enumerate(indices)
                ik, ib = ci.I
                k  = g.k_grid[ik];  b = g.b_grid[ib]
                gc  = adj_cost(k, kp, p.χ)
                res = p.R_k * k + p.R_b * b + z - kp - gc
                bp_max = max(p.b_min, min(g.b_grid[end], res - _EPS))
                bp = clamp(bp_vals[j], p.b_min, bp_max)
                k_pol[ik, ib, iz] = kp
                b_pol[ik, ib, iz] = bp
                c_pol[ik, ib, iz] = max(res - bp, _EPS)
            end
        end
    end

    return V_new, k_pol, b_pol, c_pol
end


# ============================================================
# Value Function Iteration
# ============================================================
function solve_vfi(p::Params, g::Grids;
                   tol=1e-6, max_iter=500, damping=0.4, verbose=true)
    n_k = length(g.k_grid)
    n_b = length(g.b_grid)
    n_z = length(g.z_vals)

    # Initial guess: geometric sum of hand-to-mouth utility
    c_full = Array{Float64}(undef, n_k, n_b, n_z)
    for iz in 1:n_z
        z = g.z_vals[iz]
        for ib in 1:n_b, ik in 1:n_k
            c_full[ik, ib, iz] = max(
                (p.R_k - 1.0) * g.k_grid[ik] + p.R_b * g.b_grid[ib] + z,
                _EPS
            )
        end
    end
    V = [u(c_full[ik, ib, iz], p.σ) / (1.0 - p.β)
         for ik in 1:n_k, ib in 1:n_b, iz in 1:n_z]

    # b_floor: lower bound covering all possible b* values across (k, k') pairs
    b_floor = (g.b_grid[1]
               + (p.R_k / p.R_b) * (g.k_grid[1] - g.k_grid[end])
               - adj_cost(g.k_grid[end], g.k_grid[1], p.χ) / p.R_b
               - 3.0)

    # Pre-allocate policy arrays so they survive past the loop
    k_pol = zeros(n_k, n_b, n_z)
    b_pol = zeros(n_k, n_b, n_z)
    c_pol_out = copy(c_full)

    for iter in 1:max_iter
        V_NA_spl, bp_NA_spl = egm_no_adjust(V, c_full, p, g, b_floor)
        V_new, k_pol, b_pol, c_new = maximise_over_kprime(V_NA_spl, bp_NA_spl, p, g)

        if damping > 0.0
            V_new = (1.0 - damping) .* V_new .+ damping .* V
        end

        err = maximum(abs.(V_new .- V))
        verbose && iter % 10 == 0 && println("Iter $iter, ||ΔV|| = $err")

        if err < tol
            verbose && println("Converged in $iter iterations (err = $err)")
            return (; V=V_new, k_pol, b_pol, c_pol=c_new)
        end

        V         = V_new
        c_full    = c_new
        c_pol_out = c_new

        if iter == max_iter
            @warn "VFI did not converge after $max_iter iterations (err = $err)"
        end
    end

    return (; V, k_pol, b_pol, c_pol=c_pol_out)
end


# ============================================================
# Run
# ============================================================
p = Params()
g = make_grids(p; n_k=30, n_b=50, k_max=10.0, b_max=10.0)

println("Solving two-asset model via Graves' two-step method...")
@time sol = solve_vfi(p, g; tol=1e-6, max_iter=1000, damping=0.3)
println("Done.")

println("V range:  [$(minimum(sol.V)),  $(maximum(sol.V))]")
println("c range:  [$(minimum(sol.c_pol)), $(maximum(sol.c_pol))]")
println("k' range: [$(minimum(sol.k_pol)), $(maximum(sol.k_pol))]")
println("b' range: [$(minimum(sol.b_pol)), $(maximum(sol.b_pol))]")

using Plots

ib_slices = [1, length(g.b_grid) ÷ 2, length(g.b_grid)]
labels_b  = ["b=$(round(g.b_grid[ib], digits=1))" for ib in ib_slices]

p1 = plot(title="k' policy", xlabel="k")
p2 = plot(title="b' policy", xlabel="k")
p3 = plot(title="Consumption", xlabel="k")
for (ib, lab) in zip(ib_slices, labels_b)
    for iz in 1:length(g.z_vals)
        ls = iz == 1 ? :solid : :dash
        lbl = "$lab, z=$(g.z_vals[iz])"
        plot!(p1, g.k_grid, sol.k_pol[:, ib, iz]; label=lbl, linestyle=ls)
        plot!(p2, g.k_grid, sol.b_pol[:, ib, iz]; label=lbl, linestyle=ls)
        plot!(p3, g.k_grid, sol.c_pol[:, ib, iz]; label=lbl, linestyle=ls)
    end
end
plot(p1, p2, p3; layout=(1, 3), size=(1200, 400))
#savefig("two_step_policies.png")
println("Policy plots saved.")
