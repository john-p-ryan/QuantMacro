#=
This file contains code for solving the Krusell-Smith model
with a constant unemployment transition matrix (no regime switching), 
using the Endogenous Grid Method for the household solution, and 
Young's histogram method for distributional estimates.
=#

# import packages and usefull functions
using Parameters, Random, Statistics
using Spline # custom spline package

# Steady state without aggregate risk, used for the initial distribution
module Aiyagari
    using Parameters, LinearAlgebra, Optim, SparseArrays, Spline
    include(joinpath(@__DIR__, "..", "incomplete_markets", "Aiyagari_EGM.jl"))
end


################################################################################
# Part I - Primitives, Results, and Helper Functions
################################################################################


@with_kw struct Primitives
    β::Float64 = 0.99
    γ::Float64 = 1.0  # coefficient of relative risk aversion (γ = 1 is log utility)
    @assert γ > 0 "γ must be positive"
    α::Float64 = 0.36
    δ::Float64 = 0.025
    ē::Float64 = 0.3271

    z_grid::Vector{Float64} = [1.01, 0.99]
    z_g::Float64 = z_grid[1]
    z_b::Float64 = z_grid[2]
    nz::Int64 = length(z_grid)
    ρ::Float64 = 0.75       # TFP persistence, the same default as the BKM, Reiter, and SSJ solutions
    @assert abs(ρ) < 1 "ρ must be in (-1, 1)"
    # Symmetric two-state chain whose autocorrelation, 2p - 1 with p the probability of staying in a state,
    # equals ρ, so the discretized process has the same persistence as the linear methods'
    Mzz::Matrix{Float64} = [(1+ρ)/2 (1-ρ)/2;
                            (1-ρ)/2 (1+ρ)/2]

    ϵ_grid::Vector{Float64} = [1.0, 0.0]
    nϵ::Int64 = length(ϵ_grid)
    M_unemp::Matrix{Float64} = [0.9624 0.0376; 0.5 0.5]
    unemp::Float64 = M_unemp[1, 2] / (M_unemp[1, 2] + M_unemp[2, 1])
    L::Float64 = ē * (1.0 - unemp)

    # Grid defaults match the Aiyagari solvers, whose steady state initializes KS and the linear methods
    nk::Int64 = 60
    k_min::Float64 = 1e-6
    k_max::Float64 = 60.0
    k_density::Float64 = 2.0  # curvature > 1 clusters points near k_min, where the policy functions bend
    k_grid::Vector{Float64} = make_grid(k_min, k_max, nk; density=k_density)

    # Grid for histogram method, matched to Aiyagari for initialization
    k_hist_min::Float64 = k_min
    k_hist_max::Float64 = k_max
    n_hist::Int64 = 125
    k_hist_grid::Vector{Float64} = make_grid(k_hist_min, k_hist_max, n_hist; density=1.0)

    nK::Int64 = 30
    K_min::Float64 = 10.0
    K_max::Float64 = 13.0
    K_grid::Vector{Float64} = range(K_min, stop=K_max, length=nK)
end

@with_kw mutable struct Results
    k_policy::Array{Float64, 4}
    c_policy::Array{Float64, 4}
    a₀::Float64
    a₁::Float64
    b₀::Float64
    b₁::Float64
    R²::Float64
    Z_sim_path::Vector{Float64}
    K_sim_path::Vector{Float64}
    C_sim_path::Vector{Float64}      # Aggregate consumption path
    r_sim_path::Vector{Float64}      # Interest rate path
    w_sim_path::Vector{Float64}      # Wage path
    K_var_sim_path::Vector{Float64}  # Capital variance path
end

@with_kw struct Simulations
    T::Int64 = 20_000
    seed::Int64 = 1234
    policy_tol::Float64 = 1e-9
    policy_max_iter::Int64 = 20_000
    burn::Int64 = 1000
    reg_tol::Float64 = 1e-8
    reg_max_iter::Int64 = 250
    λ::Float64 = 0.382
    threaded::Bool = true  # EGM step across Julia's threads; false runs it serially, e.g. to time methods on one core
end

# CRRA marginal utility c^(-γ); γ = 1 is log utility
function u_prime(c, γ; ε=1e-6)
    if c > ε
        return γ == 1 ? 1 / c : c^(-γ)
    else # first-order Taylor expansion around ε keeps u' finite for c ≤ 0
        return ε^(-γ) * (1 + γ * (1 - c / ε))
    end
end

function u_prime_inv(mu, γ)
    return γ == 1 ? 1 / mu : mu^(-1 / γ)
end

# foreach(f, xs), with the elements split across Julia's threads when threaded is true
function threaded_foreach(f, xs; threaded::Bool=true)
    if threaded
        Threads.@threads for x in xs
            f(x)
        end
    else
        foreach(f, xs)
    end
    return nothing
end


################################################################################
# Part II - Initialization and Shock Simulation
################################################################################


function sim_Markov(current_index::Int, M::Matrix{Float64})
    rand_num = rand()
    cumulative_sum = cumsum(M[current_index, :])
    next_index = searchsortedfirst(cumulative_sum, rand_num)
    return next_index
end

# Route flat keyword arguments to the parameter structs whose fields they name; errors on unknown names
function split_kwargs(kwargs, types...)
    unknown = setdiff(keys(kwargs), fieldnames.(types)...)
    isempty(unknown) || throw(ArgumentError("unknown parameter(s): $(join(unknown, ", "))"))
    return map(T -> (; (k => v for (k, v) in pairs(kwargs) if k in fieldnames(T))...), types)
end

# Keyword arguments override the Primitives or Simulations defaults, e.g. Initialize(; γ=2.0, nK=20, seed=1)
function Initialize(; kwargs...)
    prim_kw, sim_kw = split_kwargs(kwargs, Primitives, Simulations)
    prim = Primitives(; prim_kw...)
    sim = Simulations(; sim_kw...)

    println("Solving Aiyagari model for initial KS distribution...")
    # Solve the same household problem without aggregate risk, on the KS policy and histogram grids
    # (the Aiyagari model calls the idiosyncratic state z_grid and its transition matrix M)
    prim_ss, res_ss = Aiyagari.solve_model(;
        prim.β, prim.γ, prim.α, prim.δ, prim.ē,
        z_grid=prim.ϵ_grid,
        M=prim.M_unemp,
        k_min=prim.k_hist_min,
        k_max=prim.k_hist_max,
        nk=prim.nk,
        k_density=prim.k_density,
        n_hist=prim.n_hist
    )
    
    # Get initial distribution from the Aiyagari steady state
    initial_μ = res_ss.μ # This is a (n_hist x nϵ) matrix
    initial_K = sum(initial_μ .* prim.k_hist_grid)

    println("Aiyagari steady-state capital: $initial_K")
    println("Using Aiyagari steady-state distribution as initial condition for KS.")

    # Initialize policies: consume all cash-on-hand
    c_policy_init = zeros(prim.nk, prim.nϵ, prim.nK, prim.nz)
    k_policy_init = zeros(prim.nk, prim.nϵ, prim.nK, prim.nz)
    for z_idx in 1:prim.nz, K_idx in 1:prim.nK
        z = prim.z_grid[z_idx]
        K = prim.K_grid[K_idx]
        w = (1 - prim.α) * z * (K / prim.L)^prim.α
        r = prim.α * z * (prim.L / K)^(1 - prim.α)
        for ϵ_idx in 1:prim.nϵ, k_idx in 1:prim.nk
            budget = (1 + r - prim.δ) * prim.k_grid[k_idx] + w * prim.ē * prim.ϵ_grid[ϵ_idx]
            c_policy_init[k_idx, ϵ_idx, K_idx, z_idx] = budget
        end
    end

    # Initial law of motion: persistence 0.96 in both states, with its fixed point at the Aiyagari steady state
    a₁ = b₁ = 0.96
    a₀ = b₀ = (1 - a₁) * log(initial_K)

    res = Results(
        k_policy = k_policy_init,
        c_policy = c_policy_init,
        a₀ = a₀, a₁ = a₁, b₀ = b₀, b₁ = b₁, R² = 0.0,
        Z_sim_path = zeros(sim.T),
        K_sim_path = zeros(sim.T),
        C_sim_path = zeros(sim.T),
        r_sim_path = zeros(sim.T),
        w_sim_path = zeros(sim.T),
        K_var_sim_path = zeros(sim.T)
    )

    println("KS Initialization complete.")
    return prim, sim, res, initial_μ
end


function Bellman_EGM(prim::Primitives, res::Results; threaded::Bool=true)
    @unpack_Primitives prim
    @unpack_Results res

    k_policy_next = similar(k_policy)
    c_policy_next = similar(c_policy)

    # Create 2D interpolants for c(k,K) for each future shock state (ϵ', z')
    c_policy_itps = [
        BilinearSpline(k_grid, K_grid, c_policy[:, ϵ_prime_idx, :, z_prime_idx]; bc_type="constant")
        for ϵ_prime_idx in eachindex(ϵ_grid), z_prime_idx in eachindex(z_grid)
    ]

    # EGM is parallelizable over today's aggregate states
    threaded_foreach(CartesianIndices((nz, nK)); threaded) do zK
        z_idx, K_idx = Tuple(zK)
        z = z_grid[z_idx]
        K = K_grid[K_idx]

        # Get tomorrow's aggregate capital from the LOM
        K_prime = z == z_g ? exp(a₀ + a₁ * log(K)) : exp(b₀ + b₁ * log(K))
        K_prime = clamp(K_prime, K_min, K_max)

        # Get today's prices
        w = (1 - α) * z * (K / L)^α
        r = α * z * (L / K)^(1 - α)

        # Discounted marginal utility β (1 + r' - δ) u'(c') at every k' on the grid, for each future state (ϵ', z')
        K_prime_vec = fill(K_prime, nk)
        discounted_MU = [
            β * (1 + α * z_grid[z_prime_idx] * (L / K_prime)^(1 - α) - δ) .*
                u_prime.(evaluate_spline(c_policy_itps[ϵ_prime_idx, z_prime_idx], k_grid, K_prime_vec), γ)
            for ϵ_prime_idx in eachindex(ϵ_grid), z_prime_idx in eachindex(z_grid)
        ]

        for (ϵ_idx, ϵ) in enumerate(ϵ_grid)
            # 1. Calculate Expected Marginal Utility for each k' choice
            Euler_RHS = zeros(nk)
            for z_prime_idx in eachindex(z_grid), ϵ_prime_idx in eachindex(ϵ_grid)
                Euler_RHS .+= (Mzz[z_idx, z_prime_idx] * M_unemp[ϵ_idx, ϵ_prime_idx]) .* discounted_MU[ϵ_prime_idx, z_prime_idx]
            end

            # 2. Invert Euler to get today's consumption (off-grid)
            c_today_endog = u_prime_inv.(Euler_RHS, γ)

            # 3. Use budget constraint to get today's asset grid (endogenous)
            k_today_endog = (c_today_endog .+ k_grid .- w * ē * ϵ) ./ (1 + r - δ)

            # 4. Interpolate to get consumption policy back on the fixed grid
            c_spline = PchipSpline(k_today_endog, c_today_endog)
            c_on_grid = evaluate_spline(c_spline, k_grid)
            c_policy_next[:, ϵ_idx, K_idx, z_idx] = c_on_grid

            # 5. Get capital policy from budget constraint
            budget = (1 + r - δ) .* k_grid .+ w * ē * ϵ
            k_on_grid = budget .- c_on_grid

            # 6. Enforce borrowing constraint
            binding_indices = (k_on_grid .< k_min)
            k_on_grid[binding_indices] .= k_min
            c_policy_next[binding_indices, ϵ_idx, K_idx, z_idx] = budget[binding_indices] .- k_min
            k_policy_next[:, ϵ_idx, K_idx, z_idx] = k_on_grid
        end
    end
    return k_policy_next, c_policy_next
end

function Policy_Iteration_EGM(prim::Primitives, res::Results, sim::Simulations)
    res_next = deepcopy(res)
    error = 100 * sim.policy_tol
    iter = 0

    println("Starting KS Policy Function Iteration with EGM...")
    while error > sim.policy_tol && iter < sim.policy_max_iter
        k_next, c_next = Bellman_EGM(prim, res_next; sim.threaded)
        error = maximum(abs.(c_next - res_next.c_policy))
        res_next.k_policy, res_next.c_policy = k_next, c_next
        iter += 1
        if iter % 100 == 0 || iter == 1
            println("PFI-EGM Iteration: $iter, Policy Error: $error")
        end
    end

    println((iter == sim.policy_max_iter) ? "Max iterations reached in PFI-EGM." : "Converged in PFI-EGM after $iter iterations.")
    return res_next
end


function SimulateDistributionPath(prim::Primitives, res::Results, sim::Simulations, initial_μ::Matrix{Float64})
    @unpack_Primitives prim
    @unpack_Simulations sim
    println("Simulating KS distribution path and other aggregates...")

    # Create interpolants for capital policy (for distribution evolution)
    k_policy_itps = [
        BilinearSpline(k_grid, K_grid, res.k_policy[:, ϵ_idx, :, z_idx]; bc_type="constant")
        for ϵ_idx in 1:nϵ, z_idx in 1:nz
    ]
    
    # Create interpolants for consumption policy (for aggregate C calculation)
    c_policy_itps = [
        BilinearSpline(k_grid, K_grid, res.c_policy[:, ϵ_idx, :, z_idx]; bc_type="constant")
        for ϵ_idx in 1:nϵ, z_idx in 1:nz
    ]

    # Generate aggregate shock path
    Random.seed!(seed)
    Z_path_indices = zeros(Int, T)
    Z_path_indices[1] = 1 # Start in good state
    for t in 1:(T-1)
        Z_path_indices[t+1] = sim_Markov(Z_path_indices[t], Mzz)
    end
    Z_path = z_grid[Z_path_indices]
    
    # Initialize simulation storage
    _K_sim_path = zeros(T)
    _C_sim_path = zeros(T)
    _r_sim_path = zeros(T)
    _w_sim_path = zeros(T)
    _K_var_sim_path = zeros(T)
    
    μ_t = copy(initial_μ)
    
    # Simulation loop
    for t in 1:T
        # Aggregate capital for period t is the expectation over the distribution at the start of t
        K_t = sum(μ_t .* k_hist_grid) 
        _K_sim_path[t] = K_t

        z_idx = Z_path_indices[t]
        z_t = z_grid[z_idx]
        
        # --- Calculate aggregates for period t ---
        
        # Interest rate and wage
        _r_sim_path[t] = α * z_t * (L / K_t)^(1 - α)
        _w_sim_path[t] = (1 - α) * z_t * (K_t / L)^α
        
        # Variance of capital
        _K_var_sim_path[t] = sum(μ_t .* (k_hist_grid .- K_t).^2)

        # Aggregate Consumption
        K_t_vec = fill(K_t, n_hist)
        c_pol_hist_t = zeros(n_hist, nϵ)
        for ϵ_idx in 1:nϵ
            # Evaluate the consumption policy for (K_t, z_idx) on the k_hist_grid
            c_pol_hist_t[:, ϵ_idx] = evaluate_spline(c_policy_itps[ϵ_idx, z_idx], k_hist_grid, K_t_vec)
        end
        _C_sim_path[t] = sum(μ_t .* c_pol_hist_t)

        # Print progress
        if t % 2000 == 0 || t == 1 || t == T
            println("Simulating T=$t, K_agg=$(_K_sim_path[t])")
        end

        # --- Evolve distribution to t+1 ---
        if t < T
            μ_next = zeros(n_hist, nϵ)

            # Capital policy for (K_t, z_idx) on the k_hist_grid
            k_pol_hist_t = clamp.(reduce(hcat, [evaluate_spline(k_policy_itps[ϵ_idx, z_idx], k_hist_grid, K_t_vec)
                                                for ϵ_idx in 1:nϵ]), k_hist_min, k_hist_max)

            # Iterate over today's states (k, ϵ)
            for ϵ_idx in 1:nϵ, k_idx in 1:n_hist
                mass = μ_t[k_idx, ϵ_idx]
                if mass < 1e-12; continue; end

                k_prime = k_pol_hist_t[k_idx, ϵ_idx]

                j_high = searchsortedfirst(k_hist_grid, k_prime)

                if j_high == 1
                    j_low = 1; weight_low = 1.0; weight_high = 0.0;
                elseif j_high > n_hist
                    j_high = n_hist; j_low = n_hist; weight_low = 1.0; weight_high = 0.0;
                else
                    k_high = k_hist_grid[j_high]
                    j_low = j_high - 1
                    k_low = k_hist_grid[j_low]
                    
                    if k_high ≈ k_low; weight_low = 1.0;
                    else; weight_low = (k_high - k_prime) / (k_high - k_low); end
                    weight_high = 1.0 - weight_low
                end
                
                for ϵ_prime_idx in 1:nϵ
                    prob_trans = M_unemp[ϵ_idx, ϵ_prime_idx]
                    μ_next[j_low, ϵ_prime_idx] += weight_low * prob_trans * mass
                    if j_low != j_high
                       μ_next[j_high, ϵ_prime_idx] += weight_high * prob_trans * mass
                    end
                end
            end
            
            μ_t = μ_next / sum(μ_next)
        end
    end
    
    return _K_sim_path, _C_sim_path, _r_sim_path, _w_sim_path, _K_var_sim_path, Z_path
end

function EstimateRegression(prim::Primitives, K_path::Vector{Float64}, Z_path::Vector{Float64}, sim::Simulations)
    @unpack_Simulations sim
    println("Estimating KS regression for LOM...")
    
    reg_range = (burn + 1):(T - 1)
    Y_dep = log.(K_path[reg_range .+ 1])
    log_K_curr = log.(K_path[reg_range])
    
    is_good_dummy = (Z_path[reg_range] .== prim.z_g)
    X_indep = [ones(length(log_K_curr)) log_K_curr is_good_dummy (is_good_dummy .* log_K_curr)]

    coeffs = X_indep \ Y_dep
    b0_bad, b1_bad, b0_good_diff, b1_good_diff = coeffs

    new_a0 = b0_bad + b0_good_diff
    new_a1 = b1_bad + b1_good_diff
    new_b0 = b0_bad
    new_b1 = b1_bad

    R_sq = 1 - sum((Y_dep - X_indep * coeffs).^2) / sum((Y_dep .- mean(Y_dep)).^2)
    println("Regression: a₀=$(round(new_a0,digits=4)), a₁=$(round(new_a1,digits=4)), b₀=$(round(new_b0,digits=4)), b₁=$(round(new_b1,digits=4)), R²=$(round(R_sq,digits=4))")
    return new_a0, new_a1, new_b0, new_b1, R_sq
end

# Keyword arguments override the Primitives or Simulations defaults, e.g. SolveModel(; z_grid=[1.02, 0.98], γ=2.0)
SolveModel(; kwargs...) = SolveModel(Initialize(; kwargs...)...)

# Solves from the output of Initialize, e.g. to time the solution apart from the steady state that initializes it
function SolveModel(prim::Primitives, sim::Simulations, res::Results, initial_μ::Matrix{Float64})
    @unpack_Simulations sim

    lom_error = 100.0
    lom_iter = 0

    println("\nStarting KS main solution loop (iterating on LOM)...")
    while lom_error > sim.reg_tol && lom_iter < sim.reg_max_iter
        println("\n--- KS LOM Iteration $lom_iter ---")
        prev_coeffs = [res.a₀, res.a₁, res.b₀, res.b₁]

        res = Policy_Iteration_EGM(prim, res, sim)
        
        sim_K_path, _, _, _, _, sim_Z_path = SimulateDistributionPath(prim, res, sim, initial_μ)
        
        est_a0, est_a1, est_b0, est_b1, est_R2 = EstimateRegression(prim, sim_K_path, sim_Z_path, sim)
        new_coeffs = [est_a0, est_a1, est_b0, est_b1]

        lom_error = maximum(abs.(new_coeffs - prev_coeffs))
        println("LOM Update Error: $lom_error")

        res.a₀ = λ * est_a0 + (1 - λ) * prev_coeffs[1]
        res.a₁ = λ * est_a1 + (1 - λ) * prev_coeffs[2]
        res.b₀ = λ * est_b0 + (1 - λ) * prev_coeffs[3]
        res.b₁ = λ * est_b1 + (1 - λ) * prev_coeffs[4]
        res.R² = est_R2
        lom_iter += 1
    end

    if lom_iter == sim.reg_max_iter
        println("\nWARNING: Max iterations ($(sim.reg_max_iter)) reached in LOM loop.")
    else
        println("\nConverged in LOM loop after $lom_iter iterations.")
    end

    # Run one final simulation with converged policies and LOM to store in results
    final_K_path, final_C_path, final_r_path, final_w_path, final_K_var_path, Z_path = SimulateDistributionPath(prim, res, sim, initial_μ)
    res.K_sim_path = final_K_path
    res.C_sim_path = final_C_path
    res.r_sim_path = final_r_path
    res.w_sim_path = final_w_path
    res.K_var_sim_path = final_K_var_path
    res.Z_sim_path = Z_path

    return prim, sim, res
end


"""
    estimate_impulse_response(prim, res, sim; num_simulations=1000, max_horizon=50)

Estimates the generalized impulse response of aggregate capital (K) to a positive TFP shock, using the
law of motion for capital. Each simulation starts capital at its long-run mean and draws two TFP paths
from the same random numbers: one in the good state in period 1, one in the bad state. Without the shock,
z in period 1 would be drawn from the stationary distribution, so the response is the good path minus that
mixture, π_b (K_good - K_bad), where π_b is the stationary probability of the bad state. It is the response
to raising z from its unconditional mean to z_g. Comparing against the no-shock baseline, rather than
against the long-run mean, removes the drift that capital has from its starting point regardless of the
shock. With common random numbers the two TFP paths merge once they reach the same state, which reduces
sampling noise.

# Args
- `prim` (Primitives): The Primitives struct from the model solution.
- `res` (Results): The Results struct from the model solution.
- `sim` (Simulations): The Simulations struct for model parameters like burn-in.
- `num_simulations` (Int): The number of simulation paths to average over.
- `max_horizon` (Int): The maximum number of periods to simulate for the IRF.

# Returns
- `Tuple{Vector{Float64}, Vector{Float64}}`:
    1. The impulse response of K, as a deviation from the path without the shock. Element t is the
       capital chosen in period t, K_{t+1}; capital in period 1 is predetermined.
    2. The average path of z after a good shock in period 1 (z_t).
"""
function estimate_impulse_response(
    prim::Primitives,
    res::Results,
    sim::Simulations;
    num_simulations::Int = 1000,
    max_horizon::Int = 50
    )

    @unpack a₀, a₁, b₀, b₁ = res
    @unpack z_g, z_b, Mzz = prim
    lom(K, good) = good ? exp(a₀ + a₁ * log(K)) : exp(b₀ + b₁ * log(K))
    π_b = Mzz[1, 2] / (Mzz[1, 2] + Mzz[2, 1]) # stationary probability of the bad state

    # Start each simulation at the long-run mean of capital from the simulation results
    K_bar = mean(res.K_sim_path[(sim.burn+1):end])

    irf_K = zeros(max_horizon)
    irf_z = zeros(max_horizon)
    for _ in 1:num_simulations
        K_good, K_bad = K_bar, K_bar
        good_g, good_b = true, false # state of z on the path after a good / bad shock

        for t in 1:max_horizon
            irf_z[t] += good_g ? z_g : z_b
            K_good, K_bad = lom(K_good, good_g), lom(K_bad, good_b)
            irf_K[t] += π_b * (K_good - K_bad)

            # Next period's state is good when a shared uniform draw is below the probability of moving to it
            draw = rand()
            good_g, good_b = draw < Mzz[good_g ? 1 : 2, 1], draw < Mzz[good_b ? 1 : 2, 1]
        end
    end

    return irf_K ./ num_simulations, irf_z ./ num_simulations
end




#=
@time prim_ks, sim_ks, res_ks = SolveModel();
println("\nFinal LOM: a₀=$(res_ks.a₀), a₁=$(res_ks.a₁), b₀=$(res_ks.b₀), b₁=$(res_ks.b₁), R²=$(res_ks.R²)")
println("Final mean aggregate capital from simulation: ", mean(res_ks.K_sim_path[(sim_ks.burn+1):end]))
println("Max aggregate capital from simulation: ", maximum(res_ks.K_sim_path[(sim_ks.burn+1):end]))
println("Min aggregate capital from sumulation: ", minimum(res_ks.K_sim_path[(sim_ks.burn+1):end]))



# 1. Call the function with the solved model structs
irf_K_dev, irf_z = estimate_impulse_response(
    prim_ks, res_ks, sim_ks;
    num_simulations = 1_000_000, # Use a higher number for a smoother IRF
    max_horizon = 250
)

K_bar = mean(res_ks.K_sim_path[(sim_ks.burn+1):end])
irf_K_percent = (irf_K_dev ./ K_bar) .* 100.0
# add a 0 at the start to represent period 0 (the shock period)
irf_K_percent = vcat(0.0, irf_K_percent)
irf_K_percent = irf_K_percent[1:end-1]

println("\nEstimated Impulse Response for K (in % deviation from steady state):")
display(irf_K_percent)
println("\nEstimated Impulse Response for z (average path):")
display(irf_z)

# 3. Plot the results
using Plots
# Plot for Capital (K)
p1 = plot(1:length(irf_K_percent), irf_K_percent,
    label = "Response of K",
    ylabel = "% Deviation from S.S.",
    legend = :topright,
    linewidth = 2,
    color = :blue
)
hline!([0], linestyle=:dash, color=:black, label="", primary=false)

# Plot for TFP (z)
z_uncond_mean = (prim_ks.z_g * prim_ks.Mzz[2,1] + prim_ks.z_b * prim_ks.Mzz[1,2]) / (prim_ks.Mzz[1,2] + prim_ks.Mzz[2,1])
p2 = plot(1:length(irf_z), irf_z,
    label = "Average path of z",
    xlabel = "Quarters after shock",
    ylabel = "TFP Level",
    legend = :topright,
    linewidth = 2,
    color = :red
)
hline!([z_uncond_mean], linestyle=:dash, color=:black, label="Unconditional Mean", primary=false)


# Combine the two plots into a single figure
irf_combined_plot = plot(p1, p2, 
    layout = (2, 1), 
    plot_title = "Impulse Response to a Good TFP Shock"
)

display(irf_combined_plot)

savefig("IRF_combined.png")
=#
