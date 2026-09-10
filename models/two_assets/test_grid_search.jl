using Test
using Statistics: mean

# ── Load the module code (everything except the "Run" block) ──────
# We include only the function/struct definitions by evaluating them
# in a dedicated module so the top-level script lines don't execute.

module TwoAssetModel

using LinearAlgebra

include_string(@__MODULE__, join(readlines(joinpath(@__DIR__, "grid_search.jl"))[1:176], "\n"))

end

using .TwoAssetModel: Params, Grids, make_grids, u, adj_cost, bellman_update!, solve_vfi

# ==================================================================
# 1. Unit tests for primitives
# ==================================================================

@testset "Utility function" begin
    # CRRA with σ = 2
    @test u(1.0, 2.0) ≈ (1.0^(1 - 2) - 1) / (1 - 2)   # = 0.0
    @test u(2.0, 2.0) ≈ (2.0^(-1) - 1) / (-1)
    # Log utility (σ = 1)
    @test u(1.0, 1.0) ≈ 0.0
    @test u(exp(1), 1.0) ≈ 1.0
    # Negative / zero consumption returns large negative value
    @test u(0.0, 2.0) == -1e10
    @test u(-1.0, 2.0) == -1e10
    # Monotonicity
    @test u(2.0, 2.0) > u(1.0, 2.0)
    @test u(3.0, 2.0) > u(2.0, 2.0)
    # Concavity
    @test (u(2.0, 2.0) - u(1.0, 2.0)) > (u(3.0, 2.0) - u(2.0, 2.0))
end

@testset "Adjustment cost" begin
    # No adjustment ⟹ zero cost
    @test adj_cost(5.0, 5.0, 0.05) == 0.0
    # Symmetric in deviation
    @test adj_cost(3.0, 5.0, 0.05) ≈ adj_cost(5.0, 3.0, 0.05)
    # Positive for any adjustment
    @test adj_cost(1.0, 2.0, 0.05) > 0.0
    # Scales with χ
    @test adj_cost(1.0, 2.0, 0.10) ≈ 2 * adj_cost(1.0, 2.0, 0.05)
end

# ==================================================================
# 2. Grids construction
# ==================================================================

@testset "Grids construction" begin
    p = Params()
    g = make_grids(p; n_k = 10, n_b = 15, k_max = 5.0, b_max = 5.0)

    @test length(g.k_grid) == 10
    @test length(g.b_grid) == 15
    @test g.k_grid[1] ≈ p.k_min
    @test g.b_grid[1] ≈ p.b_min
    @test g.k_grid[end] ≈ 5.0
    @test g.b_grid[end] ≈ 5.0
    # Grids are sorted
    @test issorted(g.k_grid)
    @test issorted(g.b_grid)
    # Transition matrix rows sum to 1
    for i in 1:length(g.z_vals)
        @test sum(g.Π[i, :]) ≈ 1.0
    end
    # z_vals length matches Π dimensions
    @test size(g.Π, 1) == length(g.z_vals)
    @test size(g.Π, 2) == length(g.z_vals)
end

# ==================================================================
# 3. Budget constraint consistency
# ==================================================================

@testset "Budget constraint in consumption policy" begin
    p = Params()
    g = make_grids(p; n_k = 10, n_b = 12, k_max = 5.0, b_max = 5.0)
    sol = solve_vfi(p, g; tol = 1e-6, max_iter = 500, verbose = false)

    n_k, n_b, n_z = length(g.k_grid), length(g.b_grid), length(g.z_vals)
    for iz in 1:n_z, ib in 1:n_b, ik in 1:n_k
        k  = g.k_grid[ik]
        b  = g.b_grid[ib]
        z  = g.z_vals[iz]
        k′ = sol.k_pol[ik, ib, iz]
        b′ = sol.b_pol[ik, ib, iz]
        c  = sol.c_pol[ik, ib, iz]
        # c = R_k*k + R_b*b + z - k' - b' - g(k, k')
        c_check = p.R_k * k + p.R_b * b + z - k′ - b′ - adj_cost(k, k′, p.χ)
        @test c ≈ c_check atol = 1e-12
    end
end

# ==================================================================
# 4. Value function properties
# ==================================================================

@testset "Value function properties" begin
    p = Params()
    g = make_grids(p; n_k = 15, n_b = 20, k_max = 8.0, b_max = 8.0)
    sol = solve_vfi(p, g; tol = 1e-6, max_iter = 500, verbose = false)

    # V should be increasing in k (more wealth ⟹ higher value), holding b and z fixed
    for iz in 1:length(g.z_vals), ib in 1:length(g.b_grid)
        v_slice = sol.V[:, ib, iz]
        @test issorted(v_slice)
    end

    # V should be increasing in b, holding k and z fixed
    for iz in 1:length(g.z_vals), ik in 1:length(g.k_grid)
        v_slice = sol.V[ik, :, iz]
        @test issorted(v_slice)
    end

    # V should be higher in the high-income state (z=1.0) than low (z=0.25)
    for ib in 1:length(g.b_grid), ik in 1:length(g.k_grid)
        @test sol.V[ik, ib, 2] >= sol.V[ik, ib, 1]
    end
end

# ==================================================================
# 5. Consumption positivity
# ==================================================================

@testset "Consumption is positive" begin
    p = Params()
    g = make_grids(p; n_k = 15, n_b = 20, k_max = 8.0, b_max = 8.0)
    sol = solve_vfi(p, g; tol = 1e-6, max_iter = 500, verbose = false)

    @test all(sol.c_pol .> 0)
end

# ==================================================================
# 6. Policy functions respect constraints
# ==================================================================

@testset "Policy functions respect constraints" begin
    p = Params()
    g = make_grids(p; n_k = 15, n_b = 20, k_max = 8.0, b_max = 8.0)
    sol = solve_vfi(p, g; tol = 1e-6, max_iter = 500, verbose = false)

    # k' must be on the k_grid (≥ k_min)
    @test all(sol.k_pol .>= p.k_min)
    # b' must be on the b_grid (≥ b_min)
    @test all(sol.b_pol .>= p.b_min)
    # k' and b' should be values from the grid
    @test all(x -> x ∈ g.k_grid, sol.k_pol)
    @test all(x -> x ∈ g.b_grid, sol.b_pol)
end

# ==================================================================
# 7. Bellman operator is a contraction (one-step test)
# ==================================================================

@testset "Bellman operator contracts" begin
    p = Params()
    g = make_grids(p; n_k = 10, n_b = 12, k_max = 5.0, b_max = 5.0)
    n_k, n_b, n_z = length(g.k_grid), length(g.b_grid), length(g.z_vals)

    # Start from two different initial guesses
    V_a = zeros(n_k, n_b, n_z)
    V_b = ones(n_k, n_b, n_z) * 5.0

    V_a_new = similar(V_a)
    V_b_new = similar(V_b)
    pol_k = ones(Int, n_k, n_b, n_z)
    pol_b = ones(Int, n_k, n_b, n_z)

    bellman_update!(V_a_new, pol_k, pol_b, V_a, p, g)

    pol_k2 = ones(Int, n_k, n_b, n_z)
    pol_b2 = ones(Int, n_k, n_b, n_z)
    bellman_update!(V_b_new, pol_k2, pol_b2, V_b, p, g)

    dist_before = maximum(abs.(V_a .- V_b))
    dist_after  = maximum(abs.(V_a_new .- V_b_new))

    # After one Bellman step, distance should shrink by at least factor β
    @test dist_after < dist_before
    @test dist_after <= p.β * dist_before + 1e-10  # contraction with modulus β
end

# ==================================================================
# 8. VFI convergence
# ==================================================================

@testset "VFI converges" begin
    p = Params()
    g = make_grids(p; n_k = 10, n_b = 12, k_max = 5.0, b_max = 5.0)
    # Should converge without warnings
    sol = solve_vfi(p, g; tol = 1e-5, max_iter = 1000, verbose = false)

    # Value function should be finite everywhere
    @test all(isfinite, sol.V)
    @test all(isfinite, sol.c_pol)
end

# ==================================================================
# 9. Edge case: zero adjustment cost (χ = 0)
# ==================================================================

@testset "Zero adjustment cost" begin
    p = Params(χ = 0.0)
    g = make_grids(p; n_k = 10, n_b = 12, k_max = 5.0, b_max = 5.0)
    sol = solve_vfi(p, g; tol = 1e-6, max_iter = 500, verbose = false)

    @test all(isfinite, sol.V)
    @test all(sol.c_pol .> 0)
end

# ==================================================================
# 10. Comparative statics: higher β ⟹ more saving
# ==================================================================

@testset "Higher β leads to more saving" begin
    g_shared = make_grids(Params(); n_k = 12, n_b = 15, k_max = 6.0, b_max = 6.0)

    p_low  = Params(β = 0.90)
    p_high = Params(β = 0.98)

    sol_low  = solve_vfi(p_low,  g_shared; tol = 1e-5, max_iter = 500, verbose = false)
    sol_high = solve_vfi(p_high, g_shared; tol = 1e-5, max_iter = 500, verbose = false)

    # On average, more patient agents should save more (higher k' + b')
    avg_saving_low  = mean(sol_low.k_pol  .+ sol_low.b_pol)
    avg_saving_high = mean(sol_high.k_pol .+ sol_high.b_pol)

    @test avg_saving_high > avg_saving_low
end

println("\nAll tests passed!")
