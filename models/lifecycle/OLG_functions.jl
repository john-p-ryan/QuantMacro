# functions for use in Conesa_krueger_jl.ipynb
# John Ryan, October 2023
using Parameters, LinearAlgebra, Interpolations, Optim






function cubic_spline(x_grid, f_grid; extrapolation="Cubic")
    # Check for valid input
    if length(x_grid) != length(f_grid)
        error("x_grid and f_grid must have the same length")
    end
    if length(x_grid) < 3
        error("At least 3 points are required for cubic spline interpolation")
    end
    if !issorted(x_grid)
        error("x_grid must be sorted in ascending order")
    end
    if length(unique(x_grid)) != length(x_grid)
        error("x_grid must have unique values")
    end
    if extrapolation ∉ ("None", "Flat", "Linear", "Cubic")
        error("Invalid extrapolation method. Choose from 'None', 'Flat', 'Linear', or 'Cubic'")
    end

    n = length(x_grid)
    h = x_grid[2:n] - x_grid[1:n-1]  # Now handles unevenly spaced points

    # Construct the tridiagonal matrix (Not-a-Knot condition for Cubic extrapolation)
    A = zeros(n, n)
    if extrapolation == "Cubic"
        A[1, 1] = h[2]
        A[1, 2] = -(h[1] + h[2])
        A[1, 3] = h[1]
        A[n, n - 2] = h[n - 1]
        A[n, n - 1] = -(h[n - 2] + h[n - 1])
        A[n, n] = h[n - 2]
    else  # Natural boundary conditions for other extrapolation methods
        A[1, 1] = 1.0
        A[n, n] = 1.0
    end

    for i in 2:n-1
        A[i, i-1] = h[i-1]
        A[i, i] = 2 * (h[i-1] + h[i])
        A[i, i+1] = h[i]
    end

    # Construct the right-hand side vector
    b = zeros(n)
    for i in 2:n-1
        b[i] = 3 * ((f_grid[i+1] - f_grid[i]) / h[i] - (f_grid[i] - f_grid[i-1]) / h[i-1])
    end

    # Solve the system to get second derivatives (coefficients for the cubic terms)
    c = A \ b

    # Calculate coefficients for linear, quadratic, and constant terms
    a = f_grid
    d = (c[2:n] - c[1:n-1]) ./ (3 * h)
    b = (a[2:n] - a[1:n-1]) ./ h - h .* (c[2:n] + 2 * c[1:n-1]) / 3

    # Create a closure that evaluates the spline at a given point
    function f_spline(x_new)
        # Handle extrapolation
        if x_new < x_grid[1] || x_new > x_grid[end]
            if extrapolation == "None"
                error("Extrapolation is not allowed. x_new is outside the range of x_grid.")
            elseif extrapolation == "Flat"
                return x_new < x_grid[1] ? f_grid[1] : f_grid[end]
            elseif extrapolation == "Linear"
                if x_new < x_grid[1]
                    slope = b[1]
                    return f_grid[1] + slope * (x_new - x_grid[1])
                else
                    # Find the correct interval for the right boundary
                    i = n - 1
                    # Corrected slope calculation for right boundary
                    slope = b[i] + 2 * c[i] * h[i] + 3 * d[i] * h[i]^2
                    # Corrected anchor point: using the spline's value at the endpoint
                    return (a[i] + b[i] * h[i] + c[i] * h[i]^2 + d[i] * h[i]^3) + slope * (x_new - x_grid[i+1])
                end
            end  # Cubic (Not-a-Knot) handled below
        end

        # Find the correct interval
        i = max(1, min(searchsortedlast(x_grid, x_new), n - 1))

        # Evaluate the cubic polynomial
        dx = x_new - x_grid[i]
        return a[i] + b[i] * dx + c[i] * dx^2 + d[i] * dx^3
    end

    return f_spline
end






##########################################################################
# initialize model parameters and structures
##########################################################################
@with_kw struct Primitives
    β::Float64 = 0.97        # discount rate
    γ::Float64 = .42         # Cobb Douglas consumption weight
    σ::Float64 = 2.0         # CRRA utility
    α::Float64 = .36         # capital share
    δ::Float64 = 0.06        # depreciation rate
    N::Int64 = 66            # death age
    Jʳ::Int64 = 45           # age of retirement (last working age)
    n::Float64 = .011        # population growth rate
    a_min::Float64 = 0.001   # assets lower bound
    a_max::Float64 = 30.0    # assets upper bound
    na::Int64 = 500          # number of asset grid points
    a_grid::Array{Float64,1} = collect(range(a_min, length = na, stop = a_max))     #asset grid
    z_grid::Vector{Float64} = [3.0, 0.5]                                            # productivity shocks
    nz::Int64 = length(z_grid)                                                      # number of productivity shocks
    initial_dist::Vector{Float64} = [0.2037, 0.7963]                                # initial distribution of productivity
    Π::Matrix{Float64} = [0.9261 0.0739;                                            # Stochastic process for employment
                          0.0189 0.9811] 
    θ::Float64 = 0.11                                                               # social security tax rate
    η::Vector{Float64} = ef
end



#structure that holds model results
@with_kw mutable struct Results
    Vʳ::Matrix{Float64}         # value function retired
    aʳ::Matrix{Float64}         # asset policy function retired
    cʳ::Matrix{Float64}         # consumption policy function retired
    Vʷ::Array{Float64, 3}       # value function worker
    aʷ::Array{Float64, 3}       # asset policy function worker
    cʷ::Array{Float64, 3}       # consumption policy function worker
    lʷ::Array{Float64, 3}       # labor supply
    r::Float64                  # Equilibrium interest rate
    w::Float64                  # Equilibrium wage
    b::Float64                  # retirement benefits
    K::Float64                  # Capital
    L::Float64                  # Labor
    Fʳ::Matrix{Float64}         # stationary dist for retired
    Fʷ::Array{Float64, 3}       # stationary dist for worker
end

function Initialize(prim::Primitives)
    # Initialize value function and policy function
    Vʳ = zeros(prim.N - prim.Jʳ, prim.na)       # retired value function (matrix)
    aʳ = zeros(prim.N - prim.Jʳ, prim.na)       # retired asset policy function (matrix)
    cʳ = zeros(prim.N - prim.Jʳ, prim.na)       # retired consumption policy function (matrix)
    Vʷ = zeros(prim.Jʳ, prim.na, prim.nz)       # worker value function (tensor)
    aʷ = zeros(prim.Jʳ, prim.na, prim.nz)       # worker asset policy function (tensor)
    cʷ = zeros(prim.Jʳ, prim.na, prim.nz)       # worker consumption policy function (tensor)
    lʷ = zeros(prim.Jʳ, prim.na, prim.nz)       # labor supply (tensor)
    r = 0.05                                    # initial guess for interest rate
    w = 1.05                                    # initial guess for wage
    b = 0.2                                     # initial guess for retirement benefits
    K = 4.0                                     # initial guess for capital
    L = 0.3                                     # initial guess for labor
    Fʳ = zeros(prim.N - prim.Jʳ, prim.na)       # retired wealth distribution (matrix)
    Fʷ = zeros(prim.Jʳ, prim.na, prim.nz)       # worker wealth distribution (tensor)
    return Results(Vʳ, aʳ, cʳ, Vʷ, aʷ, cʷ, lʷ, r, w, b, K, L, Fʳ, Fʷ)
end









##########################################################################
# solve the household problem
##########################################################################



function V_induction(prim::Primitives, res::Results)
    @unpack_Primitives prim
    @unpack_Results res
    
    u(c, l) = (c^γ * (1-l)^(1-γ))^(1-σ) / (1-σ)

    # initialize with age N
    cʳ[end,:] = (1+r) * a_grid .+ b
    Vʳ[end,:] = u.(cʳ[end,:], 0)
    

    # Value function iteration for retired first using EGM
    for j in (N-Jʳ-1):-1:1  # age = j + J, iterate backwards
        # find c from Euler equation using c'
        c_today = cʳ[j+1, :] * (β * (1+r)) ^ (1 / (γ*(1-σ) -1))
        # find a from budget constraint
        a_today = (c_today + a_grid .- b) / (1+r)
        
        # interpolate consumption policy function to get back on grid
        #c_itp = linear_interpolation(a_today, c_today, extrapolation_bc=Interpolations.Line())
        c_itp = cubic_spline(a_today, c_today)
        cʳ[j, :] = c_itp.(a_grid)
        # saving policy
        aʳ[j, :] = (1+r) * a_grid .+ b - cʳ[j, :]
        # value function 
        #V_itp = linear_interpolation(a_grid, Vʳ[j+1, :], extrapolation_bc=Interpolations.Flat())
        V_itp = cubic_spline(a_grid, Vʳ[j+1, :])
        Vʳ[j, :] = u.(cʳ[j, :], 0) + β * V_itp.(aʳ[j, :])
    end


    #V_itp = linear_interpolation(a_grid, Vʳ[1, :], extrapolation_bc=Interpolations.Flat())
    V_itp = cubic_spline(a_grid, Vʳ[1, :])
    # initialize with age Jʳ
    for (z_index, z) in enumerate(z_grid)
        coef = z * η[end] * w * (1-θ)
        
        for (a_index, a) in enumerate(a_grid)
            budget = (1+r)*a + coef

            function obj(ap)
                l = (γ*coef - (1-γ)*(a*(1+r)-ap))/ coef               # labor supply given a'
                l = max(0, min(1, l))                                         # enforce bounds
                c = (1+r)*a + coef*l - ap                             # consumption given a'
                if c >= 0                                                     # check for positivity
                    val = u(c,l) + β*V_itp(ap)                                # compute value function
                    return -val
                else
                    return Inf
                end
            end

            opt = optimize(obj, a_min, budget)

            # if converged, update policy & value functions
            if opt.converged
                Vʷ[end, a_index, z_index] = -opt.minimum
                aʷ[end, a_index, z_index] = opt.minimizer
                lʷ[end, a_index, z_index] = max(0, min(1, (γ*coef - (1-γ)*(a*(1+r)-opt.minimizer)) / coef))
                cʷ[end, a_index, z_index] = (1+r)*a + coef*lʷ[end, a_index, z_index] - opt.minimizer
            else
                println("Optimization did not converge at age Jʳ with a = $a, z = $z")
            end
        end
    end


    # Value function iteration for workers
    for j in (Jʳ-1):-1:1
        # loop through age

        # interpolate value function - this is using 2 types
        #Vh_itp = linear_interpolation(a_grid, Vʷ[j+1, :, 1], extrapolation_bc=Interpolations.Flat())
        #Vl_itp = linear_interpolation(a_grid, Vʷ[j+1, :, 2], extrapolation_bc=Interpolations.Flat())
        Vh_itp = cubic_spline(a_grid, Vʷ[j+1, :, 1])
        Vl_itp = cubic_spline(a_grid, Vʷ[j+1, :, 2])
        
        for (z_index, z) in enumerate(z_grid)
            coef = z * η[j] * w * (1-θ)
            
            for (a_index, a) in enumerate(a_grid)
                budget = (1+r)*a + coef

                function obj(ap)
                    l = (γ * coef - (1-γ) * (a*(1+r)-ap)) / coef      # labor supply given a'
                    l = max(0, min(1, l))                                         # enforce bounds
                    c = (1+r) * a + coef * l - ap                             # consumption given a'
                    EV_next = Π[z_index, :] ⋅ [Vh_itp(ap), Vl_itp(ap)]     # expected value function
                    if c >= 0                                                     # check for positivity
                        val = u(c,l) + β*EV_next                                  # compute value function
                        return -val
                    else
                        return Inf
                    end
                end

                opt = optimize(obj, a_min, budget)

                if opt.converged
                    Vʷ[j, a_index, z_index] = -opt.minimum
                    aʷ[j, a_index, z_index] = opt.minimizer
                    lʷ[j, a_index, z_index] = max(0, min(1, (γ*coef - (1-γ)*(a*(1+r)-opt.minimizer)) / coef))
                    cʷ[j, a_index, z_index] = (1+r)*a + coef*lʷ[j, a_index, z_index] - opt.minimizer
                else
                    println("Optimization did not converge at age $j with a = $a, z = $z")
                end
            end
        end
    end

    return Results(Vʳ, aʳ, cʳ, Vʷ, aʷ, cʷ, lʷ, r, w, b, K, L, Fʳ, Fʷ)
end



function steady_dist(prim::Primitives, res::Results) 
    @unpack_Primitives prim
    @unpack_Results res

    Fʳ = zeros(N-Jʳ, na)
    Fʷ = zeros(Jʳ, na, nz)

    μⱼ = ones(N)
    for i in 2:N
        μⱼ[i] = μⱼ[i-1] /(1+n)
    end
    μⱼ = μⱼ/sum(μⱼ)

    # take initial dist of productivity and age 1
    Fʷ[1,1,:] = initial_dist * μⱼ[1]

    # iterate F forward through age using policy functions for workers
    for j in 1:(Jʳ-1)
        for z_index in eachindex(z_grid)
            for a_index in eachindex(a_grid)
                a_prime = aʷ[j, a_index, z_index]
                for zp_index in eachindex(z_grid)
                    if a_prime <= a_min 
                        Fʷ[j+1, 1, zp_index] += Fʷ[j, a_index, z_index] * Π[z_index, zp_index] / (1+n)
                    elseif a_prime >= a_max 
                        Fʷ[j+1, end, zp_index] += Fʷ[j, a_index, z_index] * Π[z_index, zp_index] / (1+n)
                    else
                        # find 2 nearest grid points in a_grid to a_prime
                        idx_high = searchsortedfirst(a_grid, a_prime)
                        a_high = a_grid[idx_high]
                        idx_low = idx_high - 1
                        a_low = a_grid[idx_low]

                        # split the probability between points based on distance
                        weight_high = (a_prime - a_low) / (a_high - a_low)
                        weight_low = 1 - weight_high
                        Fʷ[j+1, idx_low, zp_index] += Fʷ[j, a_index, z_index] * Π[z_index, zp_index] * weight_low / (1+n)
                        Fʷ[j+1, idx_high, zp_index] += Fʷ[j, a_index, z_index] * Π[z_index, zp_index] * weight_high / (1+n)
                    end
                end
            end
        end
    end

    # take dist of initial retired from last period of employed
    for a_index in eachindex(a_grid)
        for z_index in eachindex(z_grid)
            a_prime = aʷ[end, a_index, z_index]
            if a_prime <= a_min
                Fʳ[1, 1] += Fʷ[end, a_index, z_index] / (1+n)
            elseif a_prime >= a_max
                Fʳ[1, end] += Fʷ[end, a_index, z_index] / (1+n)
            else
                # find 2 nearest grid points in a_grid to a_prime
                idx_high = searchsortedfirst(a_grid, a_prime)
                a_high = a_grid[idx_high]
                idx_low = idx_high - 1
                a_low = a_grid[idx_low]

                # split the probability between points based on distance
                weight_high = (a_prime - a_low) / (a_high - a_low)
                weight_low = 1 - weight_high
                Fʳ[1, idx_low] += Fʷ[end, a_index, z_index] * weight_low / (1+n)
                Fʳ[1, idx_high] += Fʷ[end, a_index, z_index] * weight_high / (1+n)
            end
        end
    end

    # iterate F forward through age using policy functions for retired
    for j in 1:(N-Jʳ-1)
        for a_index in eachindex(a_grid)
            a_prime = aʳ[j, a_index]
            if a_prime <= a_min
                Fʳ[j+1, 1] += Fʳ[j, a_index] / (1+n)
            elseif a_prime >= a_max
                Fʳ[j+1, end] += Fʳ[j, a_index] / (1+n)
            else
                # find 2 nearest grid points in a_grid to a_prime
                idx_high = searchsortedfirst(a_grid, a_prime)
                a_high = a_grid[idx_high]
                idx_low = idx_high - 1
                a_low = a_grid[idx_low]

                # split the probability between points based on distance
                weight_high = (a_prime - a_low) / (a_high - a_low)
                weight_low = 1 - weight_high
                Fʳ[j+1, idx_low] += Fʳ[j, a_index] * weight_low / (1+n)
                Fʳ[j+1, idx_high] += Fʳ[j, a_index] * weight_high / (1+n)
            end
        end
    end

    return Results(Vʳ, aʳ, cʳ, Vʷ, aʷ, cʷ, lʷ, r, w, b, K, L, Fʳ, Fʷ)
end




function K_L(prim::Primitives, res::Results)
    # compute capital and labor supply implied by household decisions
    @unpack_Primitives prim
    @unpack_Results res

    # compute capital and labor
    K = 0.0
    L = 0.0
    for j in 1:Jʳ # workers
        for (a_index, a) in enumerate(a_grid)
            for (z_index, z) in enumerate(z_grid)
                K += Fʷ[j, a_index, z_index] * a
                L += Fʷ[j, a_index, z_index] * z * η[j] * lʷ[j, a_index, z_index]
            end
        end
    end

    for j in 1:(N-Jʳ)
        for (a_index, a) in enumerate(a_grid)
            K += Fʳ[j, a_index] * a
        end
    end

    return K, L
end



function market_clearing(prim::Primitives, res::Results)
    # solve for the steady state using initial guess of capital and labor

    @unpack_Primitives prim
    @unpack_Results res

    function obj(x)
        K, L = x

        if K < 0 || L < 0
            return Inf
        end

        res_copy = deepcopy(res)

        # use firm FOC to get r and w
        res_copy.r = α * (L / K) ^ (1-α) - δ
        res_copy.w = (1-α) * (K / L) ^ α
        res_copy.b = θ * res_copy.w * L / sum(res.Fʳ)

        res_copy = V_induction(prim, res_copy)
        res_copy = steady_dist(prim, res_copy)
        K_supplied, L_supplied = K_L(prim, res_copy)

        return (K - K_supplied)^2 + (L - L_supplied)^2
    end

    opt = optimize(obj, [K, L])
    if opt.g_converged
        K, L = opt.minimizer
        res_copy = deepcopy(res)
        res_copy.r = α * (L / K) ^ (1-α) - δ
        res_copy.w = (1-α) * (K / L) ^ α
        res_copy.b = θ * res_copy.w * L / sum(res.Fʳ)
        res_copy.K = K
        res_copy.L = L
        res_copy = V_induction(prim, res_copy)
        res_copy = steady_dist(prim, res_copy)
        return res_copy
    else
        println("Optimization did not converge in market_clearing")

    end
end