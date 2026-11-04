#=
This file contains skeleton code for solving the Krusell-Smith model.

Table of Contents:
1. Setup model
    - 1.1. Primitives struct
    - 1.2. Results struct
2. Generate shocks
    - 2.1. Shocks struct
    - 2.2. Simulations struct
    - 2.3. functions to generate shocks
    - 2.4. Initialization
3. Solve HH problem
    - 3.1 utility function
    - 3.2 interpolation helper function
    - 3.3 Bellman operator
    - 3.4 VFI algorithm
4. Solve model
    - 4.1 Simulate capital path
    - 4.2 Estimate regression
    - 4.3 Solve model
=#

using LinearAlgebra, Random, Interpolations, Optim, Statistics









######################### Part 1 - setup model #########################

@kwdef struct Primitives
    β::Float64 = 0.99
    γ::Float64 = 1.0  # coefficient of relative risk aversion (γ = 1 is log utility)
    α::Float64 = 0.36
    δ::Float64 = 0.025
    ē::Float64 = 0.3271

    Z_grid::Vector{Float64} = [1.01, .99]
    Z_g::Float64 = Z_grid[1]
    Z_b::Float64 = Z_grid[2]
    nZ::Int64 = length(Z_grid)

    ϵ_grid::Vector{Float64} = [1, 0]
    nϵ::Int64 = length(ϵ_grid)

    # Grid defaults match the other Aiyagari and aggregate uncertainty solvers. The k grid is uniform, not
    # curved like the other methods', because bilinear_interp requires evenly spaced grids.
    nk::Int64 = 60
    k_min::Float64 = 1e-6
    k_max::Float64 = 60.0
    k_grid::Vector{Float64} = range(k_min, stop=k_max, length=nk)

    nK::Int64 = 30
    K_min::Float64 = 10.0
    K_max::Float64 = 13.0
    K_grid::Vector{Float64} = range(K_min, stop=K_max, length=nK)

end


mutable struct Results
    Z::Vector{Float64}                      # aggregate shocks
    E::Matrix{Float64}                      # employment shocks

    V::Array{Float64, 4}                    # value function, dims (k, ϵ, K, Z)
    k_policy::Array{Float64, 4}             # capital policy, similar to V

    a₀::Float64                             # constant for capital LOM, good times
    a₁::Float64                             # coefficient for capital LOM, good times
    b₀::Float64                             # constant for capital LOM, bad times
    b₁::Float64                             # coefficient for capital LOM, bad times
    R²::Float64                             # R² for capital LOM

    K_path::Vector{Float64}                 # path of capital
    k_panel::Matrix{Float64}                # panel of individual capital holdings

end










######################### Part 2 - generate shocks #########################


@kwdef struct Shocks
    #parameters of transition matrix:
    d_ug::Float64 = 1.5 # Unemp Duration (Good Times)
    u_g::Float64 = 0.04 # Fraction Unemp (Good Times)
    d_g::Float64 = 8.0  # Duration (Good Times)
    u_b::Float64 = 0.1  # Fraction Unemp (Bad Times)
    d_b::Float64 = 8.0  # Duration (Bad Times)
    d_ub::Float64 = 2.5 # Unemp Duration (Bad Times)

    #transition probabilities for aggregate states
    pgg::Float64 = (d_g-1.0)/d_g        # probability of staying in good times
    pgb::Float64 = 1.0 / d_b            # probability of going from bad to good times
    pbg::Float64 = 1.0 / d_g            # probability of going from good to bad times
    pbb::Float64 = (d_b-1.0)/d_b        # probability of staying in bad times

    #transition probabilities for aggregate states and staying unemployed
    pgg00::Float64 = (d_ug-1.0)/d_ug
    pbb00::Float64 = (d_ub-1.0)/d_ub
    pbg00::Float64 = 1.25*pbb00
    pgb00::Float64 = 0.75*pgg00

    #transition probabilities for aggregate states and becoming unemployed
    pgg01::Float64 = (u_g - u_g*pgg00)/(1.0-u_g)
    pbb01::Float64 = (u_b - u_b*pbb00)/(1.0-u_b)
    pbg01::Float64 = (u_b - u_g*pbg00)/(1.0-u_g)
    pgb01::Float64 = (u_g - u_b*pgb00)/(1.0-u_b)

    #transition probabilities for aggregate states and becoming employed
    pgg10::Float64 = 1.0 / d_ug
    pbb10::Float64 = 1.0 / d_ub
    pbg10::Float64 = 1.0 - 1.25*pbb00
    pgb10::Float64 = 1.0 - 0.75*pgg00

    #transition probabilities for aggregate states and staying employed
    pgg11::Float64 = 1.0 - (u_g - u_g*pgg00)/(1.0-u_g)
    pbb11::Float64 = 1.0 - (u_b - u_b*pbb00)/(1.0-u_b)
    pbg11::Float64 = 1.0 - (u_b - u_g*pbg00)/(1.0-u_g)
    pgb11::Float64 = 1.0 - (u_g - u_b*pgb00)/(1.0-u_b)

    # Markov Transition Matrix
    Mgg::Array{Float64,2} = [pgg11 pgg01
                            pgg10 pgg00]

    Mbg::Array{Float64,2} = [pgb11 pgb01
                            pgb10 pgb00]

    Mgb::Array{Float64,2} = [pbg11 pbg01
                            pbg10 pbg00]

    Mbb::Array{Float64,2} = [pbb11 pbb01
                            pbb10 pbb00]

    # Joint transition of (Z, ϵ); rows and columns ordered (g, employed), (g, unemployed), (b, employed), (b, unemployed)
    M::Array{Float64,2} = [pgg*Mgg pbg*Mgb
                          pgb*Mbg pbb*Mbb]

    # aggregate transition matrix
    MZZ::Array{Float64,2} = [pgg pbg
                            pgb pbb]
end


@kwdef struct Simulations
    T::Int64 = 11_000           # number of periods to simulate
    N::Int64 = 5_000            # number of agents to simulate
    seed::Int64 = 1234          # seed for random number generator

    V_tol::Float64 = 1e-8       # tolerance for value function iteration
    V_max_iter::Int64 = 10_000  # maximum number of iterations for value function

    burn::Int64 = 1_000         # number of periods to burn for regression
    reg_tol::Float64 = 1e-7     # tolerance for regression coefficients
    reg_max_iter::Int64 = 250   # maximum number of iterations for regression
    λ::Float64 = 0.382          # update parameter for regression coefficients

    K_initial::Float64 = 11.62   # initial aggregate capital
end



function sim_markov(current_index::Int64, M::Matrix{Float64})
    #=
    Simulate the next state index given the current state index and Markov transition matrix

    Args
    current_index (Int): index current state
    M (Matrix): Markov transition matrix, rows must sum to 1
    
    Returns
    next_index (Int): next state index
    =#
    
    # Generate a random number between 0 and 1
    rand_num = rand()

    # Get the cumulative sum of the probabilities in the current row
    cumulative_sum = cumsum(M[current_index, :])

    # Find the next state index based on the random number
    next_index = searchsortedfirst(cumulative_sum, rand_num)

    return next_index
end


function draw_shocks(prim::Primitives, sho::Shocks, sim::Simulations)
    #=
    Generate a sequence of aggregate shocks and panel of idiosyncratic shocks

    Args
    prim (Primitives): model parameters
    sho (Shocks): shock parameters
    sim (Simulations): simulation parameters

    Returns
    Z (Vector): vector of aggregate shocks, length T
    E (Matrix): matrix of employment shocks, size N x T
    =#
    (; T, N, seed) = sim
    Random.seed!(seed)

    Z = zeros(T)
    E = zeros(N, T)

    # --- Initialize states for t=1 ---
    local_Z_index = 1 # Start in good state
    Z[1] = prim.Z_grid[local_Z_index]

    local_ϵ_index =  ones(Int64, N)
    num_unemp = round(Int64, N * sho.u_g)
    local_ϵ_index[1:num_unemp] .= 2 # Set initial unemployed
    E[:, 1] = [prim.ϵ_grid[i] for i in local_ϵ_index]

    # --- Loop over periods t = 2 to T ---
    # We generate state for t based on state at t-1
    for t in 2:T
        # 1. Simulate next aggregate state
        # local_Z_index currently holds the state from t-1
        next_Z_index = sim_markov(local_Z_index, sho.MZZ)
        Z[t] = prim.Z_grid[next_Z_index]

        # 2. Loop over agents to simulate employment shocks for t
        for n in 1:N
            # local_ϵ_index[n] holds the state from t-1
            current_ϵ_index_n = local_ϵ_index[n] 
            next_ϵ_index_n = 0

            if local_Z_index == 1 && next_Z_index == 1
                next_ϵ_index_n = sim_markov(current_ϵ_index_n, sho.Mgg)
            elseif local_Z_index == 1 && next_Z_index == 2
                next_ϵ_index_n = sim_markov(current_ϵ_index_n, sho.Mgb)
            elseif local_Z_index == 2 && next_Z_index == 1
                next_ϵ_index_n = sim_markov(current_ϵ_index_n, sho.Mbg)
            else # local_Z_index == 2 && next_Z_index == 2
                next_ϵ_index_n = sim_markov(current_ϵ_index_n, sho.Mbb)
            end
            
            # Store shock for period t
            E[n, t] = prim.ϵ_grid[next_ϵ_index_n] 
            # Update state vector for next iteration (t+1)
            local_ϵ_index[n] = next_ϵ_index_n 
        end

        # Update aggregate state for next iteration (t+1)
        local_Z_index = next_Z_index
    end

    return Z, E
end


function initialize(; prim::Primitives=Primitives(), sho::Shocks=Shocks(), sim::Simulations=Simulations())
    #=
    Draw shocks and allocate the results struct. Pass any parameter struct to override its defaults,
    e.g. initialize(; prim=Primitives(γ=2.0, nK=20), sim=Simulations(seed=1))
    =#
    Z, E = draw_shocks(prim, sho, sim)

    V = zeros(prim.nk, prim.nϵ, prim.nK, prim.nZ)
    k_policy = zeros(prim.nk, prim.nϵ, prim.nK, prim.nZ)

    a₀ = 0.094
    a₁ = 0.965
    b₀ = 0.085
    b₁ = 0.965
    R² = 0.0

    K_path = zeros(sim.T)
    k_panel = zeros(sim.N, sim.T)
    res = Results(Z, E, V, k_policy, a₀, a₁, b₀, b₁, R², K_path, k_panel)

    return prim, sho, sim, res
end












######################### Part 3 - HH Problem #########################


function u(c::Float64, γ::Float64; ε::Float64 = .0001)
    #=
    CRRA utility, normalized as (c^(1-γ) - 1) / (1-γ) so that it nests log utility at γ = 1.
    expm1 keeps the normalization accurate for γ near 1.
    =#
    if c < ε # a linear approximation stitching function
        # ensures obj is smooth and defined for optimization when c is very small
        return u(ε, γ; ε=ε) + (c - ε) / ε^γ  # slope u'(ε) = ε^(-γ)
    elseif γ == 1
        return log(c)
    else
        return expm1((1 - γ) * log(c)) / (1 - γ)
    end
end


function bilinear_interp(F::Array{Float64, 2}, x1::Vector{Float64}, x2::Vector{Float64})
    #=
    helper for bilinear interpolation for 2D grid with flat extrapolation

    Args
    F (Array): 2D grid of function values
    x1 (Vector): grid points for first dimension - must be evenly spaced
    x2 (Vector): grid points for second dimension - must be evenly spaced

    Returns
    interp (Function): bilinear interpolation function
    =#
    x1_grid = range(minimum(x1), maximum(x1), length=length(x1))
    x2_grid = range(minimum(x2), maximum(x2), length=length(x2))

    interp = interpolate(F, BSpline(Linear()))
    extrap = extrapolate(interp, Interpolations.Flat())
    return scale(extrap, x1_grid, x2_grid)
end


function bellman(prim::Primitives, res::Results, sho::Shocks)
    #= 
    Solve the Bellman equation for the household problem. HH state variables are (k, ϵ, K, Z).

    Args
    prim (Primitives): model parameters
    res (Results): results struct
    sho (Shocks): shock parameters

    Returns
    V_next (Array): updated value function
    k_next (Array): updated capital policy function
    =#

    (; β, γ, α, δ, ē, Z_grid, Z_g, Z_b, nZ, ϵ_grid, nϵ, k_grid, nk, K_grid, nK) = prim
    (; V, a₀, a₁, b₀, b₁) = res
    (; u_g, u_b, M) = sho

    V_next = zeros(nk, nϵ, nK, nZ)
    k_next = zeros(nk, nϵ, nK, nZ)

    # interpolate value function to solve continuously
    Vg1_interp = bilinear_interp(V[:, 1, :, 1], k_grid, K_grid)
    Vg0_interp = bilinear_interp(V[:, 2, :, 1], k_grid, K_grid)
    Vb1_interp = bilinear_interp(V[:, 1, :, 2], k_grid, K_grid)
    Vb0_interp = bilinear_interp(V[:, 2, :, 2], k_grid, K_grid)

    # begin the main loop
    for (Z_index, Z) in enumerate(Z_grid)

        # L is determined by Z by law of large numbers
        if Z == Z_g
            L = ē * (1-u_g)
        elseif Z == Z_b
            L = ē * (1-u_b)
        end
            
        for (K_index, K) in enumerate(K_grid)
            # use LOM to determine K' given K & Z (assigned once, so the closure below can capture it without boxing)
            K_prime = Z == Z_g ? exp(a₀ + a₁*log(K)) : exp(b₀ + b₁*log(K))

            w = (1-α) * Z * (K/L) ^ α
            r = α * Z * (L/K) ^ (1-α)
            
            for (ϵ_index, ϵ) in enumerate(ϵ_grid)
                M_index = ϵ_index + nϵ*(Z_index-1)
                p = M[M_index, :]

                for (k_index, k) in enumerate(k_grid)
                    budget = (1 + r - δ)*k + w*ē*ϵ

                    function obj(k_prime::Float64)
                        c = budget - k_prime

                        EV_next = p[1]*Vg1_interp(k_prime, K_prime) + 
                                  p[2]*Vg0_interp(k_prime, K_prime) + 
                                  p[3]*Vb1_interp(k_prime, K_prime) + 
                                  p[4]*Vb0_interp(k_prime, K_prime)

                        return -(u(c, γ) + β*EV_next)
                    end

                    opt = optimize(obj, 0.0, budget)

                    if Optim.converged(opt)
                        V_next[k_index, ϵ_index, K_index, Z_index] = -opt.minimum
                        k_next[k_index, ϵ_index, K_index, Z_index] = opt.minimizer
                    else
                        print("error at k_index: ", k_index, " ϵ_index: ", ϵ_index, " K_index: ", K_index, " Z_index: ", Z_index)
                        error("Optimization did not converge")
                    end
                end
            end
        end    
    end

    return V_next, k_next
end


function vfi(prim::Primitives, res::Results, sho::Shocks, sim::Simulations)
    #=
    Iterate on the value function until convergence

    Args
    prim (Primitives): model parameters
    res (Results): results struct
    sho (Shocks): shock parameters
    sim (Simulations): simulation parameters

    Returns
    res_next (Results): updated results struct

    =#
    (; V_tol, V_max_iter) = sim
    res_next = deepcopy(res)

    error = 100 * V_tol
    iter = 0

    while error > V_tol && iter < V_max_iter
        V_next, k_next = bellman(prim, res_next, sho)
        # relative criterion: for γ > 1, V near c ≈ 0 is so large that an absolute tolerance is below machine precision
        error = maximum(abs.(V_next .- res_next.V) ./ (1 .+ abs.(res_next.V)))
        res_next.V = V_next
        res_next.k_policy = k_next
        iter += 1
        # println("iter: ", iter)
        # println("error: ", error)
    end

    if iter == V_max_iter
        println("Maximum iterations reached in vfi")
    elseif error < V_tol
        println("Converged in vfi after $iter iterations")
    end

    return res_next
end













########################### Part 4 - Solve model ###########################


function simulate_capital_path(prim::Primitives, res::Results, sim::Simulations)
    #=
    Simulate the path of K on the pseudopanel using interpolated policy functions

    Args
    prim (Primitives): model parameters
    res (Results): results struct
    sim (Simulations): simulation parameters

    Returns
    K_path (Vector): path of capital
    k_panel (Matrix): panel of individual capital holdings
    =#
    (; Z_g, Z_b, k_grid, K_grid) = prim
    (; Z, E, k_policy) = res
    (; T, N, K_initial) = sim

    kpg1_interp = bilinear_interp(k_policy[:, 1, :, 1], k_grid, K_grid)
    kpg0_interp = bilinear_interp(k_policy[:, 2, :, 1], k_grid, K_grid)
    kpb1_interp = bilinear_interp(k_policy[:, 1, :, 2], k_grid, K_grid)
    kpb0_interp = bilinear_interp(k_policy[:, 2, :, 2], k_grid, K_grid)

    k_panel = zeros(N, T)
    # assume agents start with initial guess of capital
    k_panel[:, 1] .= K_initial

    K_path = zeros(T)
    K_path[1] = K_initial

    for (t, Z_t) in enumerate(Z[1:end-1])
        for n in 1:N
            if Z_t == Z_g
                if E[n, t] == 1
                    k_panel[n, t+1] = kpg1_interp(k_panel[n, t], K_path[t])
                else
                    k_panel[n, t+1] = kpg0_interp(k_panel[n, t], K_path[t])
                end
            elseif Z_t == Z_b
                if E[n, t] == 1
                    k_panel[n, t+1] = kpb1_interp(k_panel[n, t], K_path[t])
                else
                    k_panel[n, t+1] = kpb0_interp(k_panel[n, t], K_path[t])
                end
            end
        end

        K_path[t+1] = mean(k_panel[:, t+1])
    end

    return K_path, k_panel
end


function estimate_regression(prim::Primitives, res::Results, sim::Simulations)
    #=
    Estimate the law of motion for capital using simulated path

    Args
    prim (Primitives): model parameters
    res (Results): results struct
    sim (Simulations): simulation parameters

    Returns
    a₀ (Float): constant for capital LOM, good times
    a₁ (Float): coefficient for capital LOM, good times
    b₀ (Float): constant for capital LOM, bad times
    b₁ (Float): coefficient for capital LOM, bad times
    R² (Float): R² for capital LOM
    =#
    (; burn) = sim

    # trim simulated path for burn-in
    K_lag = res.K_path[burn-1:end-1]
    K_path = res.K_path[burn:end]
    Z = res.Z[burn-1:end-1]

    D = Z .== prim.Z_g # dummy variable for good times

    # DGP: log(K_t+1) = D(a₀ + a₁*log(K_t)) + (1-D)(b₀ + b₁*log(K_t)) + ε

    Y = log.(K_path)
    X = [ones(length(K_lag)) log.(K_lag) D D.*log.(K_lag)]
    b₀, b₁, β_d, β_dx = X\Y 
    a₀ = b₀ + β_d
    a₁ = b₁ + β_dx

    Y_hat = X * [b₀; b₁; β_d; β_dx]
    SS_tot = sum((Y .- mean(Y)).^2)
    SS_res = sum((Y .- Y_hat).^2)
    R² = 1 - SS_res / SS_tot

    return a₀, a₁, b₀, b₁, R²

end


function solve_model(; prim::Primitives=Primitives(), sho::Shocks=Shocks(), sim::Simulations=Simulations())
    #=
    Solve the Krusell-Smith model by iterating on the law of motion for capital

    Args
    prim, sho, sim: parameter structs, e.g. solve_model(; prim=Primitives(γ=2.0, nk=100))

    Returns
    prim, sho, sim, res: parameter structs and results struct
    =#

    prim, sho, sim, res = initialize(; prim, sho, sim)
    (; reg_tol, reg_max_iter, λ) = sim

    error = 100 * reg_tol
    iter = 0

    while error > reg_tol && iter < reg_max_iter
        res = vfi(prim, res, sho, sim)
        res.K_path, res.k_panel = simulate_capital_path(prim, res, sim)
        a₀, a₁, b₀, b₁, R² = estimate_regression(prim, res, sim)
        error = maximum(abs.([a₀ - res.a₀, a₁ - res.a₁, b₀ - res.b₀, b₁ - res.b₁, R² - res.R²]))
        res.a₀ = λ*a₀ + (1-λ)*res.a₀
        res.a₁ = λ*a₁ + (1-λ)*res.a₁
        res.b₀ = λ*b₀ + (1-λ)*res.b₀
        res.b₁ = λ*b₁ + (1-λ)*res.b₁
        res.R² = R²
        iter += 1
        println("iter: ", iter)
        println("error: ", error)
    end

    if iter == reg_max_iter
        println("Maximum iterations reached in solve_model")
    elseif error < reg_tol
        println("Converged in solve_model after $iter iterations")
    end

    return prim, sho, sim, res
end
