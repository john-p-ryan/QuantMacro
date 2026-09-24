include(joinpath(@__DIR__, "grid_search_model.jl"))

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