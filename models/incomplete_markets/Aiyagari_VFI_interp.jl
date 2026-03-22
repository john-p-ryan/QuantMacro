# This file contains Julia code to solve the Aiyagari model using interpolated VFI.


@with_kw struct Primitives
    β::Float64 = 0.99
    α::Float64 = 0.36
    δ::Float64 = 0.025
    ē::Float64 = 0.3271

    z_grid::Vector{Float64} = [1.0, 0.0]
    nz::Int64 = length(z_grid)
    M::Matrix{Float64} = [0.9624 0.0376; 0.5 0.5]
    unemp::Float64 = M[1, 2] / (M[1, 2] + M[2, 1])
    L = ē * (1 - unemp) # Fixed in Aiyagari because no aggregate uncertainty & exogenous labor supply

    k_grid::Vector{Float64} = make_grid(1e-6, 60.0, 100; density=2.0)
    k_min::Float64 = minimum(k_grid)
    k_max::Float64 = maximum(k_grid)
    nk::Int64 = length(k_grid)

    k_hist::Vector{Float64} = make_grid(k_min, k_max, 250; density=1.0)
    n_hist::Int64 = length(k_hist)
end


@with_kw mutable struct Results
    k_policy::Array{Float64, 2}            # capital policy
    c_policy::Array{Float64, 2}            # consumption policy
    V::Array{Float64, 2}                   # value function
    V_splines::Vector{PchipSplineInterpolation} # splines for value function
    k_splines::Vector{PchipSplineInterpolation} # splines for savings policy function
    k_pol_hist::Array{Float64, 2}          # capital policy on histogram grid
    T_star::SparseMatrixCSC{Float64, Int}

    K::Float64                            # aggregate capital
    w::Float64                            # wage
    r::Float64                            # interest rate
    μ::Array{Float64, 2}                   # distribution of agents

    C::Float64                            # aggregate consumption
    Y::Float64                            # aggregate output
    K_var::Float64                        # variance of capital holdings
    K_cv::Float64                         # coefficient of variation of capital
    C_var::Float64                        # variance of consumption
    C_cv::Float64                         # coefficient of variation of consumption
end

function u(c; ε=1e-7)
    if c < ε
        return log(ε) + (c-ε)/ε
    else 
        return log(c)
    end
end


function initialize(;k_min=1e-6, k_max=60.0, nk=100, n_hist=125)
    prim = Primitives(k_min=k_min, k_max=k_max, nk=nk, n_hist=n_hist,
                      k_grid=make_grid(k_min, k_max, nk; density=2.0),
                      k_hist=make_grid(k_min, k_max, n_hist; density=1.0))
    @unpack_Primitives prim

    K = 11.6
    w = (1-α) * (K / L) ^ α
    r = α * (L / K) ^ (1-α)

    k_policy = zeros(nk, nz)
    c_policy = zeros(nk, nz)
    for (z_index, z) in enumerate(z_grid)
        c_policy[:, z_index] = (1+r-δ) * k_grid .+ w * ē * z
    end
    V = zeros(nk, nz)
    k_splines = [PchipSpline(k_grid, k_policy[:, z_index]) for z_index in 1:nz]
    V_splines = [PchipSpline(k_grid, V[:, z_index]) for z_index in 1:nz]
    k_pol_hist = zeros(n_hist, nz)

    # Initialize a sparse matrix. It's much more efficient in memory.
    # Each column corresponds to a starting state, each row to an ending state.
    # T_star[j, i] = probability of transitioning from state i to state j.
    N_states = n_hist * nz
    T_star = spzeros(N_states, N_states)

    μ = ones(n_hist, nz) / (n_hist * nz)

    C, Y, K_var, K_cv, C_var, C_cv = zeros(6)

    res = Results(k_policy, c_policy, V, V_splines, k_splines, k_pol_hist, T_star, K, w, r, μ, C, Y, K_var, K_cv, C_var, C_cv)
    return prim, res
end


@inline function EV_at_kprime(p::AbstractVector{<:Real},
                              V_splines::Vector, k′::Float64)
    s = 0.0
    @inbounds for zp in eachindex(p)
        s += p[zp] * evaluate_spline(V_splines[zp], [k′])[1]  # scalar eval
    end
    return s
end



function bellman(prim::Primitives, res::Results)
    @unpack_Primitives prim
    @unpack_Results res

    V_next = similar(V)
    k_next = similar(k_policy)
    c_next = similar(c_policy)
    
    for (z_index, z) in enumerate(z_grid)
        p = M[z_index, :]
        for (k_index, k) in enumerate(k_grid)
            budget = (1+r-δ) * k + w * ē * z
            obj(kp) = -(u(budget - kp) + β * EV_at_kprime(p, V_splines, kp))

            # minimize obj with optimizer
            lower = if k_index == 1
                k_min
            else
                k_policy[k_index - 1, z_index] # monotonicity: next k' must be >= previous k'
            end
            upper = min(budget, k_max)
            result = optimize(obj, lower, upper; 
                            rel_tol=1e-10, abs_tol=1e-10)
            if result.converged
                k_next[k_index, z_index] = result.minimizer
                c_next[k_index, z_index] = budget - result.minimizer
                V_next[k_index, z_index] = -result.minimum
            else
                println("Optimization failed at (k,z) = ($k, $z)")
                # fall back to previous policy
                k_next[k_index, z_index] = k_policy[k_index, z_index]
                c_next[k_index, z_index] = c_policy[k_index, z_index]
                V_next[k_index, z_index] = V[k_index, z_index] 
            end
        end
    end
    
    V_spl_next = [PchipSpline(prim.k_grid, V_next[:, z_index]) for z_index in eachindex(prim.z_grid)]
    return V_next, k_next, c_next, V_spl_next
end



function VFI!(prim, res; tol=1e-8, max_iter=10_000)

    error = 100 * tol
    iter = 0

    while error > tol && iter < max_iter
        V_next, k_next, c_next, V_spl_next = bellman(prim, res)
        error = maximum(abs.(V_next - res.V))
        res.k_policy, res.c_policy, res.V, res.V_splines = k_next, c_next, V_next, V_spl_next
        iter += 1
    end

    if iter == max_iter
        println("Maximum iterations reached in VFI")
    elseif error < tol
        println("Converged in VFI after $iter iterations")
    end

    if any(res.c_policy .< 0)
        println("Warning: Negative consumption in policy function")
    end
    k_spl = [PchipSpline(prim.k_grid, res.k_policy[:, z_index]) for z_index in eachindex(prim.z_grid)]
    res.k_splines = k_spl
    k_pol_hist = zeros(prim.n_hist, prim.nz)
    for z_index in eachindex(prim.z_grid)
        k_pol_hist[:, z_index] = evaluate_spline(k_spl[z_index], prim.k_hist)
    end
    res.k_pol_hist = k_pol_hist
end



function build_T_star(prim::Primitives, res::Results)
    # build the sparse matrix for the T* operator
    @unpack_Primitives prim
    @unpack_Results res

    N_states = n_hist * nz
    T_star = spzeros(N_states, N_states)

    # Loop over all possible starting states (k_idx, z_idx)
    for z_idx in 1:nz
        for k_idx in 1:n_hist
            # This is the flattened index for the starting state
            source_idx = (z_idx - 1) * n_hist + k_idx

            # Get the policy-determined next-period capital
            k_prime = k_pol_hist[k_idx, z_idx]

            # Loop over all possible ending shock states z_next_idx
            for z_next_idx in 1:nz
                prob_z_transition = M[z_idx, z_next_idx]

                if k_prime <= k_min
                    # Agent is at the borrowing constraint, moves to the first grid point
                    dest_k_idx_low = 1
                    weight_low = 1.0

                    dest_idx_low = (z_next_idx - 1) * n_hist + dest_k_idx_low
                    T_star[dest_idx_low, source_idx] = prob_z_transition * weight_low
                elseif k_prime >= k_max
                    # Agent is at the top, moves to the last grid point
                    dest_k_idx_high = n_hist
                    weight_high = 1.0

                    dest_idx_high = (z_next_idx - 1) * n_hist + dest_k_idx_high
                    T_star[dest_idx_high, source_idx] = prob_z_transition * weight_high
                else
                    # Linear interpolation to find weights for two adjacent grid points
                    idx_high = searchsortedfirst(k_hist, k_prime)
                    idx_low = idx_high - 1
                    
                    k_high = k_hist[idx_high]
                    k_low = k_hist[idx_low]

                    weight_high = (k_prime - k_low) / (k_high - k_low)
                    weight_low = 1.0 - weight_high

                    # Update the transition matrix for both possible destination k points
                    dest_idx_low = (z_next_idx - 1) * n_hist + idx_low
                    dest_idx_high = (z_next_idx - 1) * n_hist + idx_high

                    T_star[dest_idx_low, source_idx] += prob_z_transition * weight_low
                    T_star[dest_idx_high, source_idx] += prob_z_transition * weight_high
                end
            end
        end
    end
    return T_star
end



function steady_dist!(prim, res; tol=1e-8, max_iter=20_000)
    @unpack_Primitives prim

    # 1. Build the transition matrix once
    T_star = build_T_star(prim, res)
    res.T_star = T_star

    #=
    # alternative way to compute the steady state distribution using eigenvalue decomposition:
    d, v = eigs(T_star, nev=1, which=:LM) # from Arpack
    μ_vec = real.(v[:, 1]) # Take the real part of the first eigenvector
    μ_vec ./= sum(μ_vec)   # Normalize to be a probability distribution
    res.μ = reshape(μ_vec, n_hist, nz) # reshape it back to the original distribution shape
    # problem: 
    =#

    # 2. Flatten the distribution into a vector
    μ_vec = vec(res.μ)

    error = 100 * tol
    iter = 0

    # 3. Iterate using fast matrix-vector multiplication
    while error > tol && iter < max_iter
        μ_vec_next = T_star * μ_vec
        error = maximum(abs.(μ_vec_next - μ_vec))
        μ_vec = μ_vec_next
        iter += 1
    end

    if iter == max_iter
        println("Maximum iterations reached in SteadyDist")
    elseif error < tol
        println("Converged in SteadyDist after $iter iterations")
    end

    # 4. Reshape the vector back to a matrix and store it
    # Remember to normalize to ensure it sums to 1, correcting for any minor floating point drift
    res.μ = reshape(μ_vec, n_hist, nz)
    res.μ ./= sum(res.μ)

end



function steady_state_capital!(prim, res)
    @unpack_Primitives prim

    function r_error(r)
        res.r = r
        res.K = L * (α / r) ^ (1 / (1-α))
        res.w = (1-α) * (res.K / L) ^ α
        # use coarse solutions to speed up convergence
        VFI!(prim, res; tol=1e-8, max_iter=10_000)
        steady_dist!(prim, res; tol=1e-8, max_iter=10_000)
        capital_supply = sum(res.μ, dims=2) ⋅ k_hist
        return (capital_supply - res.K)^2
    end

    opt = optimize(r_error, δ, .08, abs_tol=1e-10)
    res.r = opt.minimizer
    res.K = L * (α / res.r) ^ (1 / (1-α))
    res.w = (1-α) * (res.K / L) ^ α
    # perform one last fine iteration to get steady state
    VFI!(prim, res; tol=1e-12, max_iter=20_000)
    steady_dist!(prim, res; tol=1e-12, max_iter=20_000)

end



function calculate_aggregates!(prim, res)
    @unpack_Primitives prim

    # calculate aggregate capital supply
    K = sum(res.μ, dims=2) ⋅ k_hist
    res.K = K # Update the struct

    # calculate aggregate consumption
    c_pol_hist = (1+res.r - δ) * prim.k_hist .+ res.w * ē * repeat(prim.z_grid', prim.n_hist) .- res.k_pol_hist
    C = res.μ ⋅ c_pol_hist
    res.C = C

    # calculate aggregate output
    Y = K^prim.α * prim.L^(1-prim.α)
    res.Y = Y

    # calculate variance of capital holdings
    # 1. Get the marginal distribution of capital by summing over the productivity states.
    μ_k = sum(res.μ, dims=2)
    # 2. Dot the marginal distribution with the squared deviations of capital.
    K_var = μ_k ⋅ (prim.k_hist .- K).^2
    res.K_var = K_var
    res.K_cv = sqrt(K_var) / K
    
    # calculate variance and coefficient of variation for consumption
    C_var = res.μ ⋅ (c_pol_hist .- C).^2
    res.C_var = C_var
    res.C_cv = sqrt(C_var) / C
end



function solve_model(; k_min=1e-6, k_max=50.0, nk=60, n_hist=125)
    prim, res = initialize(k_min=k_min, k_max=k_max, nk=nk, n_hist=n_hist)
    steady_state_capital!(prim, res)
    calculate_aggregates!(prim, res)
    return prim, res
end