using LinearAlgebra

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
    b_grid::Vector{Float64}
    z_vals::Vector{Float64}
    Π::Matrix{Float64}   # transition matrix for z
end

function make_grids(p::Params;
    n_k = 30, n_b = 50, n_z = 3,
    k_max = 10.0, b_max = 10.0
)
    k_grid = range(p.k_min, k_max, length = n_k) |> collect
    b_grid = range(p.b_min, b_max, length = n_b) |> collect

    # Simple 3-state Markov for income (Tauchen-style, hard-coded)
    z_vals = [0.25, 1.0]
    Π = [0.8  0.2;
         0.2 0.8;]

    Grids(k_grid, b_grid, z_vals, Π)
end

# ============================================================
# Utility and adjustment cost
# ============================================================
function u(c, σ)
    if c <= 0.0
        return -1e10
    end
    if σ == 1.0
        return log(c)
    else
        return (c^(1 - σ) - 1) / (1 - σ)
    end
end

function adj_cost(k, k′, χ)
    return χ * (k - k′)^2
end

# ============================================================
# Bellman update (double grid search)
# ============================================================
function bellman_update!(V_new, pol_k, pol_b, V, p::Params, g::Grids)
    n_k = length(g.k_grid)
    n_b = length(g.b_grid)
    n_z = length(g.z_vals)

    for i_z in 1:n_z
        z = g.z_vals[i_z]

        # Expected value for each (k', b') — precompute
        # EV[ik', ib'] = Σ_z' Π(z, z') V(ik', ib', iz')
        EV = zeros(n_k, n_b)
        for ib′ in 1:n_b, ik′ in 1:n_k
            for iz′ in 1:n_z
                EV[ik′, ib′] += g.Π[i_z, iz′] * V[ik′, ib′, iz′]
            end
        end

        for i_b in 1:n_b
            b = g.b_grid[i_b]
            for i_k in 1:n_k
                k = g.k_grid[i_k]

                cash = p.R_k * k + p.R_b * b + z
                best_val = -Inf
                best_ik = 1
                best_ib = 1

                for ik′ in 1:n_k
                    k′ = g.k_grid[ik′]
                    resources = cash - k′ - adj_cost(k, k′, p.χ)

                    # If even the lowest b' is infeasible, skip
                    if resources - g.b_grid[1] <= 0.0
                        continue
                    end

                    for ib′ in 1:n_b
                        b′ = g.b_grid[ib′]
                        c = resources - b′
                        val = u(c, p.σ) + p.β * EV[ik′, ib′]

                        if val > best_val
                            best_val = val
                            best_ik = ik′
                            best_ib = ib′
                        end
                    end
                end

                V_new[i_k, i_b, i_z] = best_val
                pol_k[i_k, i_b, i_z] = best_ik
                pol_b[i_k, i_b, i_z] = best_ib
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
    pol_b = ones(Int, n_k, n_b, n_z)

    for iter in 1:max_iter
        bellman_update!(V_new, pol_k, pol_b, V, p, g)
        err = maximum(abs.(V_new .- V))

        if verbose && iter % 25 == 0
            println("Iter $iter, ||V_new - V|| = $err")
        end

        if err < tol
            verbose && println("Converged in $iter iterations (err = $err)")
            copy!(V, V_new)
            break
        end

        copy!(V, V_new)

        if iter == max_iter
            @warn "VFI did not converge after $max_iter iterations (err = $err)"
        end
    end

    # Convert index policies to levels
    k_pol = g.k_grid[pol_k]
    b_pol = g.b_grid[pol_b]
    c_pol = similar(V)
    for i_z in 1:n_z, i_b in 1:n_b, i_k in 1:n_k
        k, b, z = g.k_grid[i_k], g.b_grid[i_b], g.z_vals[i_z]
        k′, b′ = k_pol[i_k, i_b, i_z], b_pol[i_k, i_b, i_z]
        c_pol[i_k, i_b, i_z] = p.R_k * k + p.R_b * b + z - k′ - b′ - adj_cost(k, k′, p.χ)
    end

    (; V, k_pol, b_pol, c_pol)
end

# ============================================================
# Run
# ============================================================
p = Params()
g = make_grids(p; n_k = 250, n_b = 90, n_z = 2, k_max = 25.0, b_max = 9.0)

println("Solving two-asset model via double grid search...")
@time sol = solve_vfi(p, g; tol = 1e-6, max_iter = 500)
println("Done.")

using Plots
plot(g.k_grid, sol.k_pol[:, 1, :])
plot!(g.k_grid, sol.k_pol[:, 45, :])
plot!(g.k_grid, sol.k_pol[:, 90, :])

plot(g.k_grid, sol.b_pol[:, 1, :])
plot!(g.k_grid, sol.b_pol[:, 45, :])
plot!(g.k_grid, sol.b_pol[:, 90, :])

plot(g.k_grid, sol.V[:, 1, :])
plot!(g.k_grid, sol.V[:, 45, :])
plot!(g.k_grid, sol.V[:, 90, :])