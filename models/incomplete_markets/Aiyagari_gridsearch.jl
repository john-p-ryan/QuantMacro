# This file contains code to solve the Aiyagari model using grid search and VFI.


@with_kw struct Primitives
    β::Float64 = 0.99
    α::Float64 = 0.36
    δ::Float64 = 0.025
    ē::Float64 = 0.3271

    z_grid::Vector{Float64} = [1.0, 0.0]
    nz::Int = length(z_grid)
    M::Matrix{Float64} = [0.9624 0.0376; 0.5 0.5]
    unemp::Float64 = M[1, 2] / (M[1, 2] + M[2, 1])
    L = ē * (1 - unemp) # Fixed in Aiyagari because no aggregate uncertainty & exogenous labor supply

    k_grid::Vector{Float64} = range(start=1e-6, stop=60.0, length=500)
    k_min::Float64 = minimum(k_grid)
    k_max::Float64 = maximum(k_grid)
    nk::Int = length(k_grid)
end


@with_kw mutable struct Results
    V::Array{Float64, 2}                   # value function V(k, z)
    k_policy::Array{Float64, 2}            # capital policy, similar to V
    c_policy::Array{Float64, 2}            # consumption policy, similar to V
    T_star::SparseMatrixCSC{Float64, Int}  # transition matrix for the joint distribution

    K::Float64                             # aggregate capital
    w::Float64                             # wage
    r::Float64                             # interest rate
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


function initialize(;k_min=1e-6, k_max=60.0, nk=500)
    prim = Primitives(k_min=k_min, k_max=k_max, nk=nk,
                      k_grid=range(start=k_min, stop=k_max, length=nk))
    @unpack_Primitives prim

    K = 11.6
    w = (1-α) * (K / L) ^ α
    r = α * (L / K) ^ (1-α)

    k_policy = zeros(nk, nz)
    c_policy = zeros(nk, nz)
    for (z_index, z) in enumerate(z_grid)
        c_policy[:, z_index] = (1+r-δ) * k_grid .+ w * ē * z
    end
    V = u.(c_policy)

    T_star = spzeros(nk * nz, nk * nz) # will be filled in later

    μ = ones(nk, nz) / (nk * nz)

    C, Y, K_var, K_cv, C_var, C_cv = zeros(6)

    res = Results(V, k_policy, c_policy, T_star, K, w, r, μ, C, Y, K_var, K_cv, C_var, C_cv)
    return prim, res
end



function bellman(prim::Primitives, res::Results)
    @unpack_Primitives prim
    @unpack_Results res

    k_next = similar(k_policy)
    c_next = similar(c_policy)
    V_next = similar(V)

    EV = V * M'

    for (z_index, z) in enumerate(z_grid)
        choice_lower = 1

        for (k_index, k) in enumerate(k_grid)
            budget = (1+r-δ) * k + w * ē * z
            V_max = -Inf

            for k_next_index in choice_lower:nk
                k_next_val = k_grid[k_next_index]
                c = budget - k_next_val
                if c <= 0
                    # increasing k' further will only reduce consumption more
                    break
                end
                V_candidate = u(c) + β * EV[k_next_index, z_index]

                if V_candidate > V_max
                    V_max = V_candidate
                    k_next[k_index, z_index] = k_next_val
                    c_next[k_index, z_index] = c
                    V_next[k_index, z_index] = V_candidate
                    choice_lower = k_next_index
                end
            end
        end
    end

    return V_next, k_next, c_next
end


function VFI!(prim::Primitives, res::Results; tol=1e-8, max_iter=10_000)

    distance = 100 * tol
    iter = 0
    while distance > tol && iter < max_iter
        V_next, k_next, c_next = bellman(prim, res)
        distance = maximum(abs.(V_next - res.V))
        res.V = V_next
        res.k_policy = k_next
        res.c_policy = c_next
        iter += 1
    end

    if any(res.c_policy .< 0)
        println("Warning: Negative consumption in policy function")
    end

    if iter == max_iter
        println("Maximum iterations reached in VFI")
    elseif distance < tol
        println("Converged in VFI after $iter iterations")
    end

end



function build_T_star(prim::Primitives, res::Results)
    @unpack_Primitives prim
    @unpack_Results res 

    T_next = spzeros(nk * nz, nk * nz)

    for z_index in 1:nz
        for k_index in 1:nk
            # The policy function gives a k' that is already on the grid
            k_prime = k_policy[k_index, z_index]
            
            # Find the exact index of that k' in the grid
            # searchsortedfirst is efficient for this
            k_next_index = searchsortedfirst(k_grid, k_prime)

            # This calculation must match the ordering of vec()
            row = (z_index - 1) * nk + k_index

            for z_next_index in 1:nz
                prob = M[z_index, z_next_index]
                
                # Calculate the destination column index
                col = (z_next_index - 1) * nk + k_next_index
                
                T_next[row, col] += prob
            end
        end
    end

    return T_next
end



function steady_dist!(prim, res; tol=1e-10, max_iter=20_000)
    @unpack_Primitives prim

    # 1. Build the transition matrix once
    T_star = build_T_star(prim, res)
    res.T_star = T_star

    #=
    # alternative way to compute the steady state distribution using eigenvalue decomposition:
    d, v = eigs(T_star, nev=1, which=:LM) # from Arpack
    μ_vec = real.(v[:, 1]) # Take the real part of the first eigenvector
    μ_vec ./= sum(μ_vec)   # Normalize to be a probability distribution
    res.μ = reshape(μ_vec, n_hist, nϵ) # reshape it back to the original distribution shape
    =#

    # 2. Flatten the distribution into a vector
    μ_vec = vec(res.μ)

    # 3. Iterate using fast matrix multiplication
    distance = 100 * tol
    iter = 0
    while distance > tol && iter < max_iter
        μ_vec_next = T_star' * μ_vec
        distance = maximum(abs.(μ_vec_next - μ_vec))
        μ_vec = μ_vec_next
        iter += 1
    end

    if iter == max_iter
        println("Maximum iterations reached in SteadyDist")
    elseif distance < tol
        println("Converged in SteadyDist after $iter iterations")
    end

    # 4. Reshape the vector back to a matrix and store it
    # Normalize to ensure it sums to 1, correcting for any minor floating point drift
    res.μ = reshape(μ_vec, nk, nz)
    res.μ ./= sum(res.μ)
end



function steady_state_capital!(prim, res)
    @unpack_Primitives prim

    function r_error(r)
        res.r = r
        res.K = L * (α / r) ^ (1 / (1-α))
        res.w = (1-α) * (res.K / L) ^ α
        # use coarse solutions to speed up convergence
        VFI!(prim, res; tol=1e-8, max_iter=5_000)
        steady_dist!(prim, res; tol=1e-8, max_iter=10_000)
        capital_supply = sum(res.μ, dims=2) ⋅ k_grid
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
    # run after solving for steady state
    @unpack_Primitives prim
    @unpack_Results res

    # calculate aggregate capital supply
    K = sum(μ, dims=2) ⋅ k_grid
    res.K = K # Update the struct

    # calculate aggregate consumption
    C = μ ⋅ c_policy
    res.C = C # Update the struct

    # calculate aggregate output
    Y = K^α * L^(1-α)
    res.Y = Y

    # calculate variance of capital holdings
    # 1. Get the marginal distribution of capital by summing over the productivity states.
    μ_k = sum(μ, dims=2)
    # 2. Dot the marginal distribution with the squared deviations of capital.
    K_var = μ_k ⋅ (k_grid .- K).^2
    res.K_var = K_var
    res.K_cv = sqrt(K_var) / K
    
    # calculate variance and coefficient of variation for consumption
    C_var = μ ⋅ (c_policy .- C).^2
    res.C_var = C_var
    res.C_cv = sqrt(C_var) / C
end


function solve_model(; k_min=1e-6, k_max=50.0, nk=500)
    prim, res = initialize(k_min=k_min, k_max=k_max, nk=nk)
    steady_state_capital!(prim, res)
    calculate_aggregates!(prim, res)
    return prim, res
end

