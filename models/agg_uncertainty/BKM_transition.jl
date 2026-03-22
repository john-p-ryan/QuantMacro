# This file contains the code to solve the transition path for the Aiyagari model with aggregate uncertainty
# using the method of Boppart, Krusell, and Mitman (2018).


using Parameters, LinearAlgebra, Random, Optim

include("spline.jl")
include("Aiyagari_EGM.jl")

@with_kw struct Simulation
    # TFP parameters
    ρ::Float64 = 0.75       # TFP persistence
    σ::Float64 = 0.00661      # TFP shock standard deviation
    Z_shock::Float64 = 0.01 # Size of initial TFP shock (for MIT shock)
    
    # Transition path parameters
    T::Int64 = 350          # Time horizon for transition
end

# Create a struct for transition path results
@with_kw mutable struct TransitionResults
    # Time paths
    K_path::Vector{Float64}            # Aggregate capital path
    Z_path::Vector{Float64}            # TFP path
    r_path::Vector{Float64}            # Interest rate path
    w_path::Vector{Float64}            # Wage path
    
    # Policy functions (time and state indexed)
    k_policy_path::Array{Float64, 3}   # Capital policy: (k, ϵ, t)
    c_policy_path::Array{Float64, 3}   # Consumption policy: (k, ϵ, t)
    
    # Distribution (time indexed)
    μ_path::Array{Float64, 3}          # Distribution: (k, ϵ, t)
    
    # Splines for policy functions (time indexed)
    k_splines_path::Vector{Vector{PchipSplineInterpolation}} # Splines for k policy
    c_splines_path::Vector{Vector{PchipSplineInterpolation}} # Splines for c policy
    
    # Policy on histogram grid (time indexed)
    k_pol_hist_path::Array{Float64, 3} # Capital policy on histogram grid: (k, ϵ, t)

    # other aggregates implied by solution
    C_path::Vector{Float64}            # Aggregate consumption path
    Y_path::Vector{Float64}            # Aggregate output path
    K_var_path::Vector{Float64}        # Variance of capital path
    K_cv_path::Vector{Float64}         # Coeff. of variation of capital path
    C_var_path::Vector{Float64}        # Variance of consumption path
    C_cv_path::Vector{Float64}         # Coeff. of variation of consumption path
    C_log_mean_path::Vector{Float64}   # Mean of log consumption path
    C_log_var_path::Vector{Float64}    # Variance of log consumption path
end

# Initialize transition paths
function InitializeTransition(prim::Primitives, res_ss::Results; T=300, Z_shock=0.01)
    sim = Simulation(T=T, Z_shock=Z_shock)
    @unpack_Simulation sim
    @unpack_Primitives prim
    
    # Create TFP path (MIT shock with deterministic decay)
    Z_path = ones(T+1)
    Z_path[1] = 1.0 + Z_shock  # Initial shock
    for t in 2:T+1
        # log(Z_t) = ρ * log(Z_{t-1})
        Z_path[t] = exp(ρ * log(Z_path[t-1]))
    end
    
    # Initialize capital path - start at steady state
    K_path = fill(res_ss.K, T+1)
    
    # Calculate initial price paths
    r_path = zeros(T+1)
    w_path = zeros(T+1)
    for t in 1:T+1
        # Production function with TFP
        r_path[t] = α * Z_path[t] * (L / K_path[t])^(1-α)
        w_path[t] = (1-α) * Z_path[t] * (K_path[t] / L)^α
    end
    
    # Initialize policy functions - start with steady state policies
    k_policy_path = zeros(nk, nϵ, T+1)
    c_policy_path = zeros(nk, nϵ, T+1)
    
    # At time T+1, we assume we're back at steady state
    k_policy_path[:, :, T+1] .= res_ss.k_policy
    c_policy_path[:, :, T+1] .= res_ss.c_policy
    
    # Initialize splines
    k_splines_path = [deepcopy(res_ss.k_splines) for _ in 1:T+1]
    c_splines_path = [deepcopy(res_ss.c_splines) for _ in 1:T+1]
    
    # Initialize histogram policies
    k_pol_hist_path = zeros(n_hist, nϵ, T+1)
    k_pol_hist_path[:, :, T+1] .= res_ss.k_pol_hist
    
    # Initialize distribution
    μ_path = zeros(n_hist, nϵ, T+1)
    μ_path[:, :, 1] .= res_ss.μ  # Start at steady state distribution

    # Initialize other aggregates
    C_path = zeros(T+1)
    Y_path = zeros(T+1)
    K_var_path = zeros(T+1)
    K_cv_path = zeros(T+1)
    C_var_path = zeros(T+1)
    C_cv_path = zeros(T+1)
    C_log_mean_path = zeros(T+1)
    C_log_var_path = zeros(T+1)
    
    # Create and return the transition results
    res_tr = TransitionResults(
        K_path=K_path, Z_path=Z_path, r_path=r_path, w_path=w_path,
        k_policy_path=k_policy_path, c_policy_path=c_policy_path,
        μ_path=μ_path,
        k_splines_path=k_splines_path, c_splines_path=c_splines_path,
        k_pol_hist_path=k_pol_hist_path,
        C_path=C_path, Y_path=Y_path, 
        K_var_path=K_var_path, K_cv_path=K_cv_path,
        C_var_path=C_var_path, C_cv_path=C_cv_path,
        C_log_mean_path=C_log_mean_path, C_log_var_path=C_log_var_path
    )
    
    return sim, res_tr
end

# Modified Bellman equation for transition dynamics (backward iteration)
function BellmanTransition(prim::Primitives, res_tr::TransitionResults, sim::Simulation, t::Int64)
    # assume t is between 1 and T
    @unpack_Primitives prim
    @unpack_Simulation sim
    @unpack_TransitionResults res_tr
    
    # Prices at time t
    r = r_path[t]
    w = w_path[t]
    r_next = r_path[t+1]
    c_policy_next = c_policy_path[:, :, t+1]
    
    
    # Initialize current period's policies
    k_next = zeros(prim.nk, prim.nϵ)
    c_next = zeros(prim.nk, prim.nϵ)
    
    for (ϵ_index, ϵ) in enumerate(ϵ_grid)
        p = M[ϵ_index, :]
        
        # Calculate expected marginal utility next period
        EMU_prime = u_prime.(c_policy_next) * p
        
        # Get consumption today from Euler equation
        c_today = u_prime_inv.(β * (1+r_next-δ) * EMU_prime)
        
        # Get capital today from budget constraint
        k_today = (c_today + k_grid .- w*ē*ϵ) / (1+r-δ)
        
        # Interpolate consumption back to the grid
        c_spl_today = PchipSpline(k_today, c_today)
        c_next[:, ϵ_index] = evaluate_spline(c_spl_today, k_grid)
        
        # Calculate capital policy from budget constraint
        k_next[:, ϵ_index] = (1+r-δ) * k_grid .+ w*ē*ϵ - c_next[:, ϵ_index]
        
        # Check for binding borrowing constraint
        binding = k_next[:, ϵ_index] .< k_min
        if any(binding)
            k_next[binding, ϵ_index] .= k_min
            c_next[binding, ϵ_index] = (1+r-δ) * k_grid[binding] .+ w*ē*ϵ .- k_min
        end
    end
    
    return k_next, c_next
end

# Backward iteration for policy functions
function BackwardIteration!(prim::Primitives, res_tr::TransitionResults, sim::Simulation)
    @unpack_Primitives prim
    @unpack_Simulation sim
    
    # We iterate backwards from T to 1
    for t in T:-1:1
        # Solve the household problem at time t
        k_next, c_next = BellmanTransition(prim, res_tr, sim, t)
        
        # Update policy functions
        res_tr.k_policy_path[:, :, t] = k_next
        res_tr.c_policy_path[:, :, t] = c_next
        
        # Update splines
        k_splines = [PchipSpline(k_grid, k_next[:, ϵ_index]) for ϵ_index in eachindex(ϵ_grid)]
        c_splines = [PchipSpline(k_grid, c_next[:, ϵ_index]) for ϵ_index in eachindex(ϵ_grid)]
        
        res_tr.k_splines_path[t] = k_splines
        res_tr.c_splines_path[t] = c_splines
        
        # Update policy on histogram grid
        k_pol_hist = zeros(prim.n_hist, prim.nϵ)
        for ϵ_index in eachindex(ϵ_grid)
            k_pol_hist[:, ϵ_index] = evaluate_spline(k_splines[ϵ_index], k_hist)
        end
        res_tr.k_pol_hist_path[:, :, t] = k_pol_hist
    end
end

# Forward iteration for distribution
function ForwardIteration!(prim::Primitives, res_tr::TransitionResults, sim::Simulation)
    @unpack_Primitives prim 
    @unpack_Simulation sim
    
    # We already have μ_path[:, :, 1] as steady state
    # Now iterate forward from t=1 to T
    for t in 1:T
        μ_next = zeros(n_hist, nϵ)
        
        for ϵ_index in eachindex(prim.ϵ_grid)
            for k_index in eachindex(k_hist)
                k_prime = res_tr.k_pol_hist_path[k_index, ϵ_index, t]
                
                for ϵ_next in eachindex(prim.ϵ_grid)
                    if k_prime < k_min
                        μ_next[1, ϵ_next] += M[ϵ_index, ϵ_next] * res_tr.μ_path[k_index, ϵ_index, t]
                    elseif k_prime > k_max
                        μ_next[end, ϵ_next] += M[ϵ_index, ϵ_next] * res_tr.μ_path[k_index, ϵ_index, t]
                    else
                        # Find 2 nearest grid points in k_hist to k_prime
                        index_high = searchsortedfirst(k_hist, k_prime)
                        
                        # Handle edge case
                        if index_high == 1
                            μ_next[1, ϵ_next] += M[ϵ_index, ϵ_next] * res_tr.μ_path[k_index, ϵ_index, t]
                            continue
                        end
                        
                        k_high = k_hist[index_high]
                        index_low = index_high - 1
                        k_low = k_hist[index_low]
                        
                        # Split probability mass between points based on distance
                        weight_high = (k_prime - k_low) / (k_high - k_low)
                        weight_low = 1 - weight_high
                        
                        μ_next[index_low, ϵ_next] += weight_low * M[ϵ_index, ϵ_next] * res_tr.μ_path[k_index, ϵ_index, t]
                        μ_next[index_high, ϵ_next] += weight_high * M[ϵ_index, ϵ_next] * res_tr.μ_path[k_index, ϵ_index, t]
                    end
                end
            end
        end
        
        # Update distribution for next period
        res_tr.μ_path[:, :, t+1] = μ_next
    end
end

# Calculate aggregate capital from distribution and policy functions
function AggregateCapital(sim::Simulation, res_tr::TransitionResults)
    # Compute aggregate capital supply for each period
    K_supply = copy(res_tr.K_path)
    for t in 2:(sim.T+1)
        K_supply[t] = sum(res_tr.μ_path[:, :, t-1] .* res_tr.k_pol_hist_path[:, :, t-1]) # changed to t-1
    end
    
    return K_supply
end

# Update prices based on capital path
function UpdatePrices!(prim::Primitives, res_tr::TransitionResults, sim::Simulation)
    @unpack_Primitives prim
    @unpack_Simulation sim
    
    for t in 1:T+1
        # Update interest rate and wage using production function with TFP
        res_tr.r_path[t] = α * res_tr.Z_path[t] * (L / res_tr.K_path[t])^(1-α)
        res_tr.w_path[t] = (1-α) * res_tr.Z_path[t] * (res_tr.K_path[t] / L)^α
    end
end

# Main function to solve for transition path
function SolveTransitionPath(prim::Primitives, res_ss::Results; T=300, Z_shock=.01, tol_path=1e-12, max_iter_path=5000, ν=0.95)
    @unpack_Primitives prim
    
    # Initialize transition path
    sim, res_tr = InitializeTransition(prim, res_ss; T, Z_shock)
    @unpack_Simulation sim
    
    # Iteration counter and error
    iter = 0
    error = 100.0 * tol_path
    
    println("Starting transition path iteration...")
    
    while error > tol_path && iter < max_iter_path
        iter += 1
        
        # Step 1: Given K_path, update prices
        UpdatePrices!(prim, res_tr, sim)
        
        # Step 2: Solve household problem backwards given prices
        BackwardIteration!(prim, res_tr, sim)
        
        # Step 3: Iterate distribution forward given policy functions
        ForwardIteration!(prim, res_tr, sim)
        
        # Step 4: Calculate implied aggregate capital supply
        K_supply = AggregateCapital(sim, res_tr)
        
        # Step 5: Check convergence
        error = maximum(abs.(K_supply[2:end-1] - res_tr.K_path[2:end-1])) # error on the interior
        
        # Step 6: Update guess with convex combination
        res_tr.K_path = ν * res_tr.K_path + (1-ν) * K_supply

        # Robustness: re-anchor the path
        res_tr.K_path[1] = res_ss.K
        res_tr.K_path[T+1] = res_ss.K
        
        if iter % 10 == 0
            println("Iteration $iter: Max error = $error")
        end
    end
    
    if iter == max_iter_path
        println("Maximum iterations reached in transition path. Final error: $error")
    else
        println("Transition path converged after $iter iterations with error: $error")
    end
    
    return sim, res_tr
end

# Function to calculate aggregates from transition results
function CalculateAggregatesTransition!(prim::Primitives, res_tr::TransitionResults, sim::Simulation)
    @unpack_Primitives prim 
    @unpack_TransitionResults res_tr
    @unpack_Simulation sim

    for t in 1:sim.T+1
        # Get the consumption policy on the histogram grid for time t
        c_pol_hist = zeros(n_hist, nϵ)
        for ϵ_index in eachindex(ϵ_grid)
            c_pol_hist[:, ϵ_index] = evaluate_spline(c_splines_path[t][ϵ_index], k_hist)
        end

        # Calculate aggregate consumption and output
        C_path[t] = res_tr.μ_path[:, :, t] ⋅ c_pol_hist
        Y_path[t] = res_tr.Z_path[t] * res_tr.K_path[t]^α * L^(1-α)

        # --- Capital Moments ---
        # 1. Marginal distribution of capital at time t
        μ_k_t = vec(sum(res_tr.μ_path[:, :, t], dims=2))
        
        # 2. Variance of capital at time t
        K_var_path[t] = μ_k_t ⋅ (k_hist .- res_tr.K_path[t]).^2
        
        # 3. Coefficient of variation of capital at time t
        if res_tr.K_path[t] > 0
            K_cv_path[t] = sqrt(K_var_path[t]) / res_tr.K_path[t]
        end

        # --- Consumption Moments ---
        # 1. Variance of consumption at time t
        C_var_path[t] = res_tr.μ_path[:, :, t] ⋅ (c_pol_hist .- C_path[t]).^2
        
        # 2. Coefficient of variation of consumption at time t
        if C_path[t] > 0
            C_cv_path[t] = sqrt(C_var_path[t]) / C_path[t]
        end

        # --- Log-Consumption Moments ---
        log_c = log.(max.(c_pol_hist, 1e-12)) # Add max for numerical stability
        C_log_mean_path[t] = res_tr.μ_path[:, :, t] ⋅ log_c
        C_log_var_path[t] = res_tr.μ_path[:, :, t] ⋅ (log_c .- C_log_mean_path[t]).^2
    end

    # Assign all calculated paths back to the results struct
    res_tr.C_path = C_path
    res_tr.Y_path = Y_path
    res_tr.K_var_path = K_var_path
    res_tr.K_cv_path = K_cv_path
    res_tr.C_var_path = C_var_path
    res_tr.C_cv_path = C_cv_path
    res_tr.C_log_mean_path = C_log_mean_path
    res_tr.C_log_var_path = C_log_var_path
end


# Main function to solve the model with BKM method
function SolveModelTransition(; T=300, Z_shock=.01)
    # First solve the steady state model
    prim, res_ss = SolveModel(; k_min=1e-6, k_max = 60.0, nk=60, n_hist=125)

    # Solve for transition path
    sim, res_tr = SolveTransitionPath(prim, res_ss; T, Z_shock)

    # calculate additional aggregates
    CalculateAggregatesTransition!(prim, res_tr, sim)
    
    return prim, res_ss, sim, res_tr
end

function SolveTransition_from_SS(prim::Primitives, res_ss::Results; T=300, Z_shock=.01)
    # Initialize transition path
    sim, res_tr = InitializeTransition(prim, res_ss; T, Z_shock)

    # Solve for transition path
    sim, res_tr = SolveTransitionPath(prim, res_ss; T, Z_shock)

    # calculate additional aggregates
    CalculateAggregatesTransition!(prim, res_tr, sim)

    return sim, res_tr
end

function ComputeIRFs(prim::Primitives, res_ss::Results, res_tr::TransitionResults, sim::Simulation; use_levels::Bool=false)
    @unpack_Primitives prim
    @unpack_Results res_ss 
    @unpack_TransitionResults res_tr

    if use_levels
        # For level deviations: normalize by LOG of the shock
        # IRF represents: change in K (in levels) per unit change in log(Z)
        log_Z_shock = log(1.0 + sim.Z_shock)
        scale = 1/log_Z_shock
        
        K_change = scale * (K_path .- K)
        Z_change = scale * log.(Z_path)  # Log deviation in Z
        r_change = scale * (r_path .- r)
        w_change = scale * (w_path .- w)
        C_change = scale * (C_path .- C)
        Y_change = scale * (Y_path .- Y)
        
        K_var_change = scale * (K_var_path .- K_var)
        K_cv_change = scale * (K_cv_path .- K_cv)
        C_var_change = scale * (C_var_path .- C_var)
        C_cv_change = scale * (C_cv_path .- C_cv)
        C_log_var_change = scale * (C_log_var_path .- C_log_var)

        return (K = K_change, Z = Z_change, r = r_change, w = w_change,
                C = C_change, Y = Y_change, 
                K_var = K_var_change, K_cv = K_cv_change,
                C_var = C_var_change, C_cv = C_cv_change,
                C_log_var = C_log_var_change)
    else
        scale = 1/sim.Z_shock
        # Original: Compute percentage deviations from steady state
        K_pct_change = scale * (K_path .- K) ./ K 
        Z_pct_change = scale * (Z_path .- 1.0) 
        r_pct_change = scale * (r_path .- r) ./ r
        w_pct_change = scale * (w_path .- w) ./ w
        C_pct_change = scale * (C_path .- C) ./ C
        Y_pct_change = scale * (Y_path .- Y) ./ Y
        
        K_var_pct_change = scale * (K_var_path .- K_var) ./ K_var
        K_cv_pct_change = scale * (K_cv_path .- K_cv) ./ K_cv
        C_var_pct_change = scale * (C_var_path .- C_var) ./ C_var
        C_cv_pct_change = scale * (C_cv_path .- C_cv) ./ C_cv
        C_log_var_pct_change = scale * (C_log_var_path .- C_log_var) ./ C_log_var

        return (K = K_pct_change, Z = Z_pct_change, r = r_pct_change, w = w_pct_change,
                C = C_pct_change, Y = Y_pct_change, 
                K_var = K_var_pct_change, K_cv = K_cv_pct_change,
                C_var = C_var_pct_change, C_cv = C_cv_pct_change,
                C_log_var = C_log_var_pct_change)
    end
end


function convolve(irf::Vector{Float64}, innovations::Vector{Float64})
    T_len = length(innovations)
    deviation_path = zeros(T_len)
    irf_len = length(irf)

    for T_idx in 1:T_len
        current_deviation = 0.0
        for s in 1:T_idx
            # Effect of innovation at time `s` on the aggregate at time `T_idx`.
            # The time lag is `T_idx - s`. The corresponding IRF index is `T_idx - s + 1`.
            irf_idx = T_idx - s + 1
            if irf_idx <= irf_len
                current_deviation += irf[irf_idx] * innovations[s]
            end
        end
        deviation_path[T_idx] = current_deviation
    end
    return deviation_path
end



function simulate_economy(Z_path::Vector{Float64}, irfs, res_ss::Results, prim::Primitives, sim::Simulation; use_levels::Bool=false)
    @unpack_Primitives prim
    @unpack_Results res_ss
    @unpack_Simulation sim
    T_sim = length(Z_path)
    
    # --- Step 1: Back out the sequence of TFP innovations from the TFP path ---
    log_Z_path = log.(Z_path)
    innovations = zeros(T_sim)
    innovations[1] = log_Z_path[1] 

    for t in 2:T_sim
        innovations[t] = log_Z_path[t] - ρ * log_Z_path[t-1]
    end

    # --- Step 2: Convolve IRFs with shocks ---
    if use_levels
        # For level IRFs: convolve directly with log innovations (mean-zero!)
        # No conversion to level shocks - this avoids Jensen's inequality bias
        K_dev_path = convolve(irfs.K, innovations)
        K_var_dev_path = convolve(irfs.K_var, innovations)
        K_cv_dev_path = convolve(irfs.K_cv, innovations)
        C_dev_path = convolve(irfs.C, innovations)
        C_var_dev_path = convolve(irfs.C_var, innovations)
        C_cv_dev_path = convolve(irfs.C_cv, innovations)
        Y_dev_path = convolve(irfs.Y, innovations)
        r_dev_path = convolve(irfs.r, innovations)
        w_dev_path = convolve(irfs.w, innovations)
        
        # IRFs are in level deviations, so just add to steady state
        K_sim_path = K .+ K_dev_path
        K_var_sim_path = K_var .+ K_var_dev_path
        K_cv_sim_path = K_cv .+ K_cv_dev_path
        C_sim_path = C .+ C_dev_path
        C_var_sim_path = C_var .+ C_var_dev_path
        C_cv_sim_path = C_cv .+ C_cv_dev_path
        Y_sim_path = Y .+ Y_dev_path
        r_sim_path = r .+ r_dev_path
        w_sim_path = w .+ w_dev_path
    else
        # For percent deviations: convert to level shocks first
        level_shock_series = exp.(innovations) .- 1.0
        
        K_dev_path = convolve(irfs.K, level_shock_series)
        K_var_dev_path = convolve(irfs.K_var, level_shock_series)
        K_cv_dev_path = convolve(irfs.K_cv, level_shock_series)
        C_dev_path = convolve(irfs.C, level_shock_series)
        C_var_dev_path = convolve(irfs.C_var, level_shock_series)
        C_cv_dev_path = convolve(irfs.C_cv, level_shock_series)
        Y_dev_path = convolve(irfs.Y, level_shock_series)
        r_dev_path = convolve(irfs.r, level_shock_series)
        w_dev_path = convolve(irfs.w, level_shock_series)
        
        # IRFs are in percentage deviations, so multiply
        K_sim_path = K .* (1.0 .+ K_dev_path)
        K_var_sim_path = K_var .* (1.0 .+ K_var_dev_path)
        K_cv_sim_path = K_cv .* (1.0 .+ K_cv_dev_path)
        C_sim_path = C .* (1.0 .+ C_dev_path)
        C_var_sim_path = C_var .* (1.0 .+ C_var_dev_path)
        C_cv_sim_path = C_cv .* (1.0 .+ C_cv_dev_path)
        Y_sim_path = Y .* (1.0 .+ Y_dev_path)
        r_sim_path = r .* (1.0 .+ r_dev_path)
        w_sim_path = w .* (1.0 .+ w_dev_path)
    end
    
    return (K=K_sim_path, K_var=K_var_sim_path, K_cv=K_cv_sim_path, 
            C=C_sim_path, C_var=C_var_sim_path, C_cv=C_cv_sim_path,
            Y=Y_sim_path, r=r_sim_path, w=w_sim_path)
end



#=
prim, res_ss, sim, res_tr = SolveModelTransition(;T=1500)
irfs = ComputeIRFs(prim, res_ss, res_tr, sim)
using Plots
plot(irfs.K_var, label="Variance of Capital", title="", legend=:topright, 
    linewidth=2, xlabel="Time", ylabel="% Deviation from Steady State", dpi=300)
savefig("K_var_transition.png")
=#

#plot(irfs.K_cv, label="Coeff. of Variation of Capital", title="", legend=:topright, 
#    linewidth=2, xlabel="Time", ylabel="% Deviation from Steady State", dpi=300)

#=
prim, res_ss = SolveModel();

# 1. Define the different shock sizes you want to test
shock_sizes = [0.001, 0.01, 0.10, 0.25, 0.5] # 0.01%, 0.1%, 1%, 10%
irf_results = Dict{Float64, NamedTuple}() # Dictionary to store results

# 2. Loop through each shock size, solve the transition, and compute IRFs
for shock in shock_sizes
    println("="^50)
    println("Solving for a Z_shock of $(shock*100)%")
    
    # Solve the transition path for the given shock
    # Using SolveTransition_from_SS is efficient as it re-uses the steady state
    sim, res_tr = SolveTransition_from_SS(prim, res_ss; T=350, Z_shock=shock)
    
    # Compute and store the normalized IRFs
    irf_results[shock] = ComputeIRFs(prim, res_ss, res_tr, sim)
    
    println("Done.")
    println("="^50)
end

# 3. Plot the results to compare
# Example: Plotting the IRF for Aggregate Capital (K)
T_plot = 200 # Number of periods to plot
p = plot(title="Impulse Response of Aggregate Capital", xlabel="Quarters", ylabel="% Deviation from SS")

for shock in shock_sizes
    label_str = "Shock: $(shock*100)%"
    plot!(p, 1:T_plot, irf_results[shock].K[1:T_plot], label=label_str, lw=2)
end

display(p)
savefig(p, "K_irfs_scaled.png")
=#