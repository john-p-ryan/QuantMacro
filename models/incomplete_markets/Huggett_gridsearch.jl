# This file contains code to solve the Huggett model using grid search and VFI.


# Employment transitions with an expected unemployment spell of D periods and a long-run unemployment rate unemp
function employment_transitions(D, unemp)
    p_ue = 1 / D                       # job-finding probability
    p_eu = p_ue * unemp / (1 - unemp)  # job-loss probability that makes unemp the long-run unemployment rate
    return [1-p_eu p_eu; p_ue 1-p_ue]
end


@with_kw struct Primitives
    β::Float64 = 0.994
    γ::Float64 = 1.5  # coefficient of relative risk aversion (γ = 1 is log utility)

    b::Float64 = 0.5                    # unemployment benefit
    s_grid::Vector{Float64} = [1.0, b]  # earnings when employed and unemployed
    ns::Int = length(s_grid)
    Π::Matrix{Float64} = employment_transitions(2.0, 0.0566)  # spells last 2 periods on average, 5.66% unemployed
    unemp::Float64 = Π[1, 2] / (Π[1, 2] + Π[2, 1])          # long-run unemployment rate implied by Π

    a_min::Float64 = -2.0  # borrowing constraint
    a_max::Float64 = 4.0
    # Grid search restricts choices to this uniform grid, which is also the support of the distribution, so it
    # needs a much finer grid than EGM.
    na::Int = 500
    a_grid::Vector{Float64} = range(start=a_min, stop=a_max, length=na)
    @assert a_grid[1] ≈ a_min && a_grid[end] ≈ a_max && length(a_grid) == na "a_grid must span [a_min, a_max] with na points"
end


@with_kw mutable struct Results
    V::Array{Float64, 2}                   # value function V(a, s)
    a_policy::Array{Float64, 2}            # asset policy, similar to V
    c_policy::Array{Float64, 2}            # consumption policy, similar to V
    T_star::SparseMatrixCSC{Float64, Int}  # transition matrix for the joint distribution

    q::Float64                             # bond price
    A::Float64                             # aggregate assets (zero in equilibrium: bonds are in zero net supply)
    μ::Array{Float64, 2}                   # distribution of agents

    C::Float64                            # aggregate consumption
    A_var::Float64                        # variance of asset holdings (no coefficient of variation: mean assets are zero)
    C_var::Float64                        # variance of consumption
    C_cv::Float64                         # coefficient of variation of consumption
    cum_pop::Vector{Float64}               # Lorenz curve: cumulative population share, poorest first
    lorenz::Vector{Float64}                # Lorenz curve: cumulative wealth share
    gini::Float64                         # Gini coefficient of wealth
    W::Float64                            # welfare: expected value E[V(a, s)] under μ
end


# CRRA utility, normalized as (c^(1-γ) - 1) / (1-γ) so that it nests log utility at γ = 1.
# expm1 keeps the normalization accurate for γ near 1. Below ε, u is extended linearly.
function u(c, γ; ε=1e-7)
    if c < ε
        return u(ε, γ; ε=ε) + (c - ε) / ε^γ  # slope u'(ε) = ε^(-γ)
    elseif γ == 1
        return log(c)
    else
        return expm1((1 - γ) * log(c)) / (1 - γ)
    end
end


function initialize(prim::Primitives)
    @unpack_Primitives prim

    q = β  # initial guess for the bond price: its complete-markets level

    # initial guess: consume all resources, keeping debt at the borrowing constraint
    a_policy = fill(a_min, na, ns)
    c_policy = zeros(na, ns)
    for (s_index, s) in enumerate(s_grid)
        c_policy[:, s_index] = a_grid .+ s .- q * a_min
    end
    V = u.(c_policy, γ)

    T_star = spzeros(na * ns, na * ns) # will be filled in later

    μ = ones(na, ns) / (na * ns)

    A, C, A_var, C_var, C_cv, gini, W = zeros(7)
    cum_pop, lorenz = zeros(0), zeros(0)

    res = Results(V, a_policy, c_policy, T_star, q, A, μ, C, A_var, C_var, C_cv, cum_pop, lorenz, gini, W)
    return res
end



function bellman(prim::Primitives, res::Results)
    @unpack_Primitives prim
    @unpack_Results res

    a_next = similar(a_policy)
    c_next = similar(c_policy)
    V_next = similar(V)

    EV = V * Π'

    for (s_index, s) in enumerate(s_grid)
        choice_lower = 1

        for (a_index, a) in enumerate(a_grid)
            budget = a + s
            V_max = -Inf

            for a_next_index in choice_lower:na
                a_next_val = a_grid[a_next_index]
                c = budget - q * a_next_val
                if c <= 0
                    # increasing a' further will only reduce consumption more
                    break
                end
                V_candidate = u(c, γ) + β * EV[a_next_index, s_index]

                if V_candidate > V_max
                    V_max = V_candidate
                    a_next[a_index, s_index] = a_next_val
                    c_next[a_index, s_index] = c
                    V_next[a_index, s_index] = V_candidate
                    choice_lower = a_next_index
                end
            end
        end
    end

    return V_next, a_next, c_next
end


function VFI!(prim::Primitives, res::Results; tol=1e-8, max_iter=10_000)

    distance = 100 * tol
    iter = 0
    while distance > tol && iter < max_iter
        V_next, a_next, c_next = bellman(prim, res)
        # relative criterion: for γ > 1, V near c ≈ 0 is so large that an absolute tolerance is below machine precision
        distance = maximum(abs.(V_next .- res.V) ./ (1 .+ abs.(res.V)))
        res.V = V_next
        res.a_policy = a_next
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

    # T_star[row, col] = probability of moving from state row to state col, in the ordering of vec(μ). Collect the
    # nonzeros as (row, column, value) triplets and assemble the matrix once at the end, which is much faster than
    # inserting them into a sparse matrix one at a time.
    rows, cols, vals = Int[], Int[], Float64[]

    for s_index in 1:ns
        for a_index in 1:na
            # The policy function gives an a' that is already on the grid
            a_prime = a_policy[a_index, s_index]

            # Find the exact index of that a' in the grid
            # searchsortedfirst is efficient for this
            a_next_index = searchsortedfirst(a_grid, a_prime)

            # This calculation must match the ordering of vec()
            row = (s_index - 1) * na + a_index

            for s_next_index in 1:ns
                prob = Π[s_index, s_next_index]

                # Calculate the destination column index
                col = (s_next_index - 1) * na + a_next_index

                push!(rows, row)
                push!(cols, col)
                push!(vals, prob)
            end
        end
    end

    return sparse(rows, cols, vals, na * ns, na * ns)
end



function steady_dist!(prim, res; tol=1e-10, max_iter=20_000)
    @unpack_Primitives prim

    # 1. Build the transition matrix once
    T_star = build_T_star(prim, res)
    res.T_star = T_star

    # 2. Flatten the distribution into a vector
    μ_vec = vec(copy(res.μ))
    μ_vec_next = similar(μ_vec)

    # 3. Iterate using fast, in-place matrix-vector multiplication
    distance = 100 * tol
    iter = 0
    while distance > tol && iter < max_iter
        mul!(μ_vec_next, transpose(T_star), μ_vec)
        distance = maximum(abs.(μ_vec_next .- μ_vec))
        μ_vec, μ_vec_next = μ_vec_next, μ_vec
        iter += 1
    end

    if iter == max_iter
        println("Maximum iterations reached in SteadyDist")
    elseif distance < tol
        println("Converged in SteadyDist after $iter iterations")
    end

    # 4. Reshape the vector back to a matrix and store it
    # Normalize to ensure it sums to 1, correcting for any minor floating point drift
    res.μ = reshape(μ_vec, na, ns)
    res.μ ./= sum(res.μ)
end



function steady_state_equilibrium!(prim, res)
    @unpack_Primitives prim

    function assets_error(q)
        res.q = q
        # use coarse solutions to speed up convergence
        VFI!(prim, res; tol=1e-8, max_iter=5_000)
        steady_dist!(prim, res; tol=1e-8, max_iter=10_000)
        # bonds are in zero net supply, so the market clears when aggregate asset holdings are zero
        asset_supply = sum(res.μ, dims=2) ⋅ a_grid
        return asset_supply^2
    end

    # Precautionary saving pushes the bond price above its complete-markets level β (the interest rate below the rate
    # of time preference). The search assumes a nonnegative interest rate, which very tight constraints can violate.
    opt = optimize(assets_error, β, 1.0, abs_tol=1e-10)
    res.q = opt.minimizer
    # perform one last fine iteration to get steady state
    VFI!(prim, res; tol=1e-12, max_iter=20_000)
    steady_dist!(prim, res; tol=1e-12, max_iter=20_000)

end



function calculate_aggregates!(prim, res)
    # run after solving for steady state
    @unpack_Primitives prim
    @unpack_Results res

    # calculate aggregate assets
    μ_a = vec(sum(μ, dims=2)) # marginal distribution of assets
    A = μ_a ⋅ a_grid
    res.A = A

    # calculate aggregate consumption
    C = μ ⋅ c_policy
    res.C = C

    # calculate variance of asset holdings
    res.A_var = μ_a ⋅ (a_grid .- A).^2

    # calculate variance and coefficient of variation for consumption
    C_var = μ ⋅ (c_policy .- C).^2
    res.C_var = C_var
    res.C_cv = sqrt(C_var) / C

    # calculate Lorenz curve and Gini coefficient of wealth, defined as assets plus earnings a + s
    wealth = vec(a_grid .+ s_grid') # same ordering as vec(μ)
    order = sortperm(wealth)
    wealth_sorted = wealth[order]
    mass_sorted = vec(μ)[order]
    res.cum_pop = [0.0; cumsum(mass_sorted)]
    res.lorenz = [0.0; cumsum(wealth_sorted .* mass_sorted)] ./ (wealth_sorted ⋅ mass_sorted)
    # Gini = 1 - 2 × (area under the Lorenz curve), with the area from the trapezoid rule
    res.gini = 1.0 - sum(diff(res.cum_pop) .* (res.lorenz[2:end] .+ res.lorenz[1:end-1]))

    # calculate the welfare measure W = E[V(a, s)] under μ
    res.W = μ ⋅ V
end


# Keyword arguments override the Primitives defaults, e.g. solve_model(; a_min=-1.0, na=1000)
solve_model(; kwargs...) = solve_model(Primitives(; kwargs...))

function solve_model(prim::Primitives)
    res = initialize(prim)
    steady_state_equilibrium!(prim, res)
    calculate_aggregates!(prim, res)
    return prim, res
end
