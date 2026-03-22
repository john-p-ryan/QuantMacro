using Parameters, LinearAlgebra, Optim

@with_kw struct Primitives
    β::Float64 = 0.8                        # discount factor
    c_f::Float64 = 1.9                      # fixed cost of operation
    c_e::Float64 = 3.0                      # fixed cost of entry
    α::Float64 = 0.5                        # labor share
    φ_grid::Vector{Float64} = [.60, 1.0]    # productivity grid
    nφ::Int64 = length(φ_grid)              # number of productivity states
    F::Array{Float64,2} = [.9 .1; .1 .9]    # transition matrix
    ν::Vector{Float64} = [.5, .5]           # initial distribution
    d::Float64 = 5.0                        # demand parameter
end


@with_kw mutable struct Results
    V::Vector{Float64}                      # value function
    μ::Vector{Float64}                      # distribution of existing firms
    π_vec::Vector{Float64}                  # indirect profit function
    n_policy::Vector{Float64}               # labor demand policy
    x_policy::Vector{Int64}                 # exit policy = 1 if exit
    P::Float64                              # final goods price
    M::Float64                              # measure of entrants
    Q::Float64                              # aggregate output
end


calc_n_policy(P::Float64, φ::Float64; prim::Primitives) = (P * prim.α * φ)^(1 / (1-prim.α))

function calc_π(P::Float64, φ::Float64; prim::Primitives)
    n = calc_n_policy(P, φ; prim)
    π_vec = P*φ*n^prim.α - n - prim.c_f
    return π_vec
end


function initialize(prim::Primitives)
    V = zeros(prim.nφ)
    μ = repeat([1/prim.nφ], prim.nφ)
    P = 4.0
    M = 1.0
    Q = 1.0
    π_vec = calc_π.(P, prim.φ_grid; prim=prim)
    n_policy = calc_n_policy.(P, prim.φ_grid; prim=prim)
    x_policy = zeros(Int64, prim.nφ)
    return Results(V, μ, π_vec, n_policy, x_policy, P, M, Q)
end


function V_bellman(prim::Primitives, res::Results)
    @unpack_Primitives prim
    @unpack_Results res

    V_next = zeros(prim.nφ)
    x_next = zeros(Int64, prim.nφ)
    
    for φ_index in eachindex(φ_grid)
        E_V = V ⋅ F[φ_index,:]
        x_next[φ_index] = E_V < 0 ? 1 : 0
        V_next[φ_index] = π_vec[φ_index] + β * (1 - x_next[φ_index]) * E_V
    end

    return V_next, x_next
end


function V_iterate(prim::Primitives, res::Results; tol::Float64=1e-9, max_iter::Int64=1000)

    err = 100*tol
    iter = 0
    res_new = deepcopy(res)

    while err > tol && iter <= max_iter
        V_next, x_next = V_bellman(prim, res_new)
        err = maximum(abs.(V_next - res_new.V))
        res_new.V = V_next
        res_new.x_policy = x_next
        iter += 1
    end
    if iter == max_iter
        println("Maximum iterations reached in V_iterate")
    elseif err < tol
        # println("Converged in V_iterate after $iter iterations")
        return res_new
    end
end


function P_iterate(prim::Primitives, res::Results; tol=1e-8)
    @unpack_Primitives prim
    # solve for P that satisfies entry condition given an eqbm with entry
    res_new = deepcopy(res)
    function obj_fun(P::Float64)
        res_new.P = P
        res_new.n_policy = calc_n_policy.(P, φ_grid; prim)
        res_new.π_vec = calc_π.(P, φ_grid; prim)
        res_new = V_iterate(prim, res_new)
        EC_residual = res_new.V ⋅ ν - c_e
        return EC_residual^2
    end

    result = optimize(obj_fun, 0.0, d)
    if isapprox(result.minimum, 0.0; atol=tol) # check if optimization converged
        # println("Equilibrium P = ", result.minimizer)
        return res_new
    else
        println("Optimization did not converge in P_iterate")
    end
end


function μ_iterate(prim::Primitives, res::Results; tol=1e-8, max_iter=1000)
    @unpack_Primitives prim
    # solves for μ given P, policy functions, and M

    error = 100*tol
    iter = 0
    res_new = deepcopy(res)

    while error > tol && iter < max_iter
        μ_next = zeros(nφ)
        μ_next = F' * (res_new.μ .* (1 .- res_new.x_policy)) + res_new.M * ν
        error = maximum(abs.(μ_next - res_new.μ))
        res_new.μ = μ_next
        iter += 1
    end
    if iter == max_iter
        println("Maximum iterations reached in μ_iterate")
    elseif error < tol
        #println("Converged in μ_iterate after $iter iterations")
        return res_new
    end 
end


function M_iterate(prim::Primitives, res::Results; tol=1e-8)
    @unpack_Primitives prim
    # solve for M using market clearing given P and policy functions
    Q_d = D - res.P  # inverse demand function for P < 1/d
    res_new = deepcopy(res)

    function obj_fun(M::Float64)
        res_new.M = M
        res_new = μ_iterate(prim, res_new)
        Q_s = res_new.μ ⋅ (φ_grid .* (res_new.n_policy .^ α))
        Q_residual = Q_s - Q_d
        return Q_residual^2
    end

    result = optimize(obj_fun, 0.0, 1.0)
    if isapprox(result.minimum, 0.0; atol=tol)
        res_new.M = result.minimizer
        res_new.Q = Q_d
        #println("Equilibrium M = ", result.minimizer[1])
        return res_new
    else
        println("Optimization did not converge in M_iterate")
    end
end


function Solve_model_competitive(prim::Primitives)
    prim, res = initialize(prim)
    res = P_iterate(prim, res)
    res = M_iterate(prim, res)
    return res
end