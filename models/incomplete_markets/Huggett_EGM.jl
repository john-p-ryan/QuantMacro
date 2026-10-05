# This file contains code to solve the Huggett model using endogenous grid method (EGM) and policy iteration.


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
    ns::Int64 = length(s_grid)
    Π::Matrix{Float64} = employment_transitions(2.0, 0.0566)  # spells last 2 periods on average, 5.66% unemployed
    unemp::Float64 = Π[1, 2] / (Π[1, 2] + Π[2, 1])          # long-run unemployment rate implied by Π

    # set the grids through a_min, a_max, na, a_density, and n_hist.
    a_min::Float64 = -2.0     # borrowing constraint
    a_max::Float64 = 4.0
    na::Int64 = 100
    a_density::Float64 = 2.0  # curvature > 1 clusters points near a_min, where the policy functions bend
    a_grid::Vector{Float64} = make_grid(a_min, a_max, na; density=a_density)
    @assert a_grid[1] ≈ a_min && a_grid[end] ≈ a_max && length(a_grid) == na "a_grid must span [a_min, a_max] with na points"

    n_hist::Int64 = 250
    a_hist::Vector{Float64} = make_grid(a_min, a_max, n_hist; density=1.0)
end


@with_kw mutable struct Results
    a_policy::Array{Float64, 2}            # asset policy
    c_policy::Array{Float64, 2}            # consumption policy
    V::Array{Float64, 2}                   # value function
    a_splines::Vector{PchipSplineInterpolation} # splines for asset policy function
    c_splines::Vector{PchipSplineInterpolation} # splines for consumption function
    a_pol_hist::Array{Float64, 2}          # asset policy on histogram grid
    V_hist::Array{Float64, 2}              # value function on histogram grid
    T_star::SparseMatrixCSC{Float64, Int}

    q::Float64                            # bond price
    A::Float64                            # aggregate assets (zero in equilibrium: bonds are in zero net supply)
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

function u_prime(c, γ; ε=1e-7)
    c = max(c, ε)
    return γ == 1 ? 1 / c : c^(-γ)
end

function u_prime_inv(x, γ)
    if x > 0
        return γ == 1 ? 1 / x : x^(-1 / γ)
    else
        error("invalid marginal utility value")
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
    V = zeros(na, ns)
    a_splines = [PchipSpline(a_grid, a_policy[:, s_index]) for s_index in 1:ns]
    c_splines = [PchipSpline(a_grid, c_policy[:, s_index]) for s_index in 1:ns]
    a_pol_hist = zeros(n_hist, ns)
    V_hist = zeros(n_hist, ns)

    # Initialize a sparse matrix. It's much more efficient in memory.
    # Each column corresponds to a starting state, each row to an ending state.
    # T_star[j, i] = probability of transitioning from state i to state j.
    N_states = n_hist * ns
    T_star = spzeros(N_states, N_states)

    μ = ones(n_hist, ns) / (n_hist * ns)

    A, C, A_var, C_var, C_cv, gini, W = zeros(7)
    cum_pop, lorenz = zeros(0), zeros(0)

    res = Results(a_policy, c_policy, V, a_splines, c_splines, a_pol_hist, V_hist, T_star, q, A, μ,
                  C, A_var, C_var, C_cv, cum_pop, lorenz, gini, W)
    return res
end



function bellman(prim::Primitives, res::Results)
    @unpack_Primitives prim
    @unpack_Results res

    a_next = similar(a_policy)
    c_next = similar(c_policy)

    # expected marginal utility tomorrow at each a' on the grid, E[u'(c(a', s')) | s], one column per s
    EMU_prime = u_prime.(c_policy, γ) * Π'

    for (s_index, s) in enumerate(s_grid)
        # get consumption today (off grid) from Euler equation q u'(c) = β E[u'(c')]
        c_today = u_prime_inv.(β / q * EMU_prime[:, s_index], γ)
        # get assets today (off grid) from budget constraint c + q a' = a + s
        a_today = c_today .+ q * a_grid .- s
        # interpolate consumption policy function to get back on the grid
        c_spl_today = PchipSpline(a_today, c_today)
        c_next[:, s_index] = evaluate_spline(c_spl_today, a_grid)
        # calculate asset policy function from budget constraint
        a_next[:, s_index] = (a_grid .+ s .- c_next[:, s_index]) / q

        # The borrowing constraint binds below a_today[1], the assets at which the Euler equation holds with a' = a_min.
        # Checking a' < a_min as well guards against interpolation error just above it.
        binding = (a_grid .< a_today[1]) .| (a_next[:, s_index] .< a_min)
        if any(binding)
            a_next[binding, s_index] .= a_min
            c_next[binding, s_index] .= a_grid[binding] .+ s .- q * a_min
        end
    end

    return a_next, c_next
end



function policy_iteration!(prim, res; tol=1e-8, max_iter=10_000)

    error = 100 * tol
    iter = 0

    while error > tol && iter < max_iter
        a_next, c_next = bellman(prim, res)
        error = maximum(abs.(c_next - res.c_policy))
        res.a_policy, res.c_policy = a_next, c_next
        iter += 1
    end

    if iter == max_iter
        println("Maximum iterations reached in policy iteration")
    elseif error < tol
        println("Converged in policy iteration after $iter iterations")
    end

    c_spl = [PchipSpline(prim.a_grid, res.c_policy[:, s_index]) for s_index in eachindex(prim.s_grid)]
    res.c_splines = c_spl
    a_spl = [PchipSpline(prim.a_grid, res.a_policy[:, s_index]) for s_index in eachindex(prim.s_grid)]
    res.a_splines = a_spl
    a_pol_hist = zeros(prim.n_hist, prim.ns)
    for s_index in eachindex(prim.s_grid)
        a_pol_hist[:, s_index] = evaluate_spline(a_spl[s_index], prim.a_hist)
    end
    res.a_pol_hist = a_pol_hist
end



function build_T_star(prim::Primitives, res::Results)
    # build the sparse matrix for the T* operator
    @unpack_Primitives prim
    @unpack_Results res

    N_states = n_hist * ns

    # Collect the nonzeros as (row, column, value) triplets and assemble the matrix once at the end, which is much
    # faster than inserting them into a sparse matrix one at a time. Each state has at most 2ns successors.
    nnz_max = 2 * ns * N_states
    rows, cols, vals = sizehint!(Int[], nnz_max), sizehint!(Int[], nnz_max), sizehint!(Float64[], nnz_max)

    # Loop over all possible starting states (a_idx, s_idx)
    for s_idx in 1:ns
        for a_idx in 1:n_hist
            # This is the flattened index for the starting state
            source_idx = (s_idx - 1) * n_hist + a_idx

            # Get the policy-determined next-period assets
            a_prime = a_pol_hist[a_idx, s_idx]

            # Split a' between the two adjacent grid points with weights that preserve its mean (Young 2010)
            if a_prime <= a_min
                # Agent is at the borrowing constraint, moves to the first grid point
                idx_low, idx_high, weight_high = 1, 2, 0.0
            elseif a_prime >= a_max
                # Agent is at the top, moves to the last grid point
                idx_low, idx_high, weight_high = n_hist - 1, n_hist, 1.0
            else
                # Linear interpolation to find weights for two adjacent grid points
                idx_high = searchsortedfirst(a_hist, a_prime)
                idx_low = idx_high - 1
                weight_high = (a_prime - a_hist[idx_low]) / (a_hist[idx_high] - a_hist[idx_low])
            end

            # Loop over all possible ending employment states s_next_idx
            for s_next_idx in 1:ns
                prob_s_transition = Π[s_idx, s_next_idx]

                push!(rows, (s_next_idx - 1) * n_hist + idx_low, (s_next_idx - 1) * n_hist + idx_high)
                push!(cols, source_idx, source_idx)
                push!(vals, prob_s_transition * (1 - weight_high), prob_s_transition * weight_high)
            end
        end
    end

    # drop the zero weights stored when a' falls on a grid point
    return dropzeros!(sparse(rows, cols, vals, N_states, N_states))
end



function steady_dist!(prim, res; tol=1e-8, max_iter=20_000)
    @unpack_Primitives prim

    # 1. Build the transition matrix once
    T_star = build_T_star(prim, res)
    res.T_star = T_star

    # 2. Flatten the distribution into a vector
    μ_vec = vec(copy(res.μ))
    μ_vec_next = similar(μ_vec)

    error = 100 * tol
    iter = 0

    # 3. Iterate using fast, in-place matrix-vector multiplication
    while error > tol && iter < max_iter
        mul!(μ_vec_next, T_star, μ_vec)
        error = maximum(abs.(μ_vec_next .- μ_vec))
        μ_vec, μ_vec_next = μ_vec_next, μ_vec
        iter += 1
    end

    if iter == max_iter
        println("Maximum iterations reached in SteadyDist")
    elseif error < tol
        println("Converged in SteadyDist after $iter iterations")
    end

    # 4. Reshape the vector back to a matrix and store it
    # Remember to normalize to ensure it sums to 1, correcting for any minor floating point drift
    res.μ = reshape(μ_vec, n_hist, ns)
    res.μ ./= sum(res.μ)
end



function steady_state_equilibrium!(prim, res)
    @unpack_Primitives prim

    function assets_error(q)
        res.q = q
        # use coarse solutions to speed up convergence
        policy_iteration!(prim, res; tol=1e-9, max_iter=10_000)
        steady_dist!(prim, res; tol=1e-9, max_iter=10_000)
        # bonds are in zero net supply, so the market clears when aggregate asset holdings are zero
        asset_supply = sum(res.μ, dims=2) ⋅ a_hist
        return asset_supply^2
    end

    # Precautionary saving pushes the bond price above its complete-markets level β (the interest rate below the rate
    # of time preference). The search assumes a nonnegative interest rate, which very tight constraints can violate.
    opt = optimize(assets_error, β, 1.0, abs_tol=1e-9)
    res.q = opt.minimizer
    # perform one last fine iteration to get steady state
    policy_iteration!(prim, res; tol=1e-12, max_iter=20_000)
    steady_dist!(prim, res; tol=1e-12, max_iter=20_000)
end



function calculate_aggregates!(prim, res)
    @unpack_Primitives prim

    # calculate aggregate assets
    μ_a = vec(sum(res.μ, dims=2)) # marginal distribution of assets
    A = μ_a ⋅ a_hist
    res.A = A

    # calculate aggregate consumption
    c_pol_hist = zeros(n_hist, ns)
    for s_index in eachindex(s_grid)
        c_pol_hist[:, s_index] = evaluate_spline(res.c_splines[s_index], a_hist)
    end
    C = res.μ ⋅ c_pol_hist
    res.C = C

    # calculate variance of asset holdings
    res.A_var = μ_a ⋅ (a_hist .- A).^2

    # calculate variance and coefficient of variation for consumption
    C_var = res.μ ⋅ (c_pol_hist .- C).^2
    res.C_var = C_var
    res.C_cv = sqrt(C_var) / C

    # calculate Lorenz curve and Gini coefficient of wealth, defined as assets plus earnings a + s
    wealth = vec(a_hist .+ s_grid') # same ordering as vec(μ)
    order = sortperm(wealth)
    wealth_sorted = wealth[order]
    mass_sorted = vec(res.μ)[order]
    res.cum_pop = [0.0; cumsum(mass_sorted)]
    res.lorenz = [0.0; cumsum(wealth_sorted .* mass_sorted)] ./ (wealth_sorted ⋅ mass_sorted)
    # Gini = 1 - 2 × (area under the Lorenz curve), with the area from the trapezoid rule
    res.gini = 1.0 - sum(diff(res.cum_pop) .* (res.lorenz[2:end] .+ res.lorenz[1:end-1]))
end



function recover_V!(prim::Primitives, res::Results; tol=1e-10, max_iter=20_000)
    # Recover V from fixed policies: V(a,s) = u(c_policy(a,s)) + β * E[V(a_policy(a,s), s')]
    @unpack_Primitives prim
    @unpack_Results res


    # Good initial guess: geometric series under fixed policies
    V = if maximum(abs.(res.V)) > 0
        copy(res.V)
    else
        u.(res.c_policy, γ) ./ (1 - β)
    end

    # The policies are fixed, so flow utility and the points a' at which V is evaluated are too
    flow_u = u.(res.c_policy, γ)
    aprime = [clamp.(res.a_policy[:, s_idx], a_min, a_max) for s_idx in 1:ns]  # guard against tiny drift
    V_at_aprime = zeros(na)

    err = Inf
    iter = 0

    while err > tol && iter < max_iter
        # Build splines of current V for each s' to evaluate at off-grid a'
        V_spl = [PchipSpline(a_grid, V[:, s_next]) for s_next in 1:ns]

        V_new = similar(V)
        for s_idx in 1:ns
            # Expected continuation value E[V(a', s')]
            EV = zeros(na)
            @inbounds for sp in 1:ns
                EV .+= Π[s_idx, sp] .* evaluate_spline!(V_at_aprime, V_spl[sp], aprime[s_idx])
            end
            V_new[:, s_idx] = flow_u[:, s_idx] .+ β .* EV
        end

        # relative criterion: for γ > 1, V near c ≈ 0 is so large that an absolute tolerance is below machine precision
        err = maximum(abs.(V_new .- V) ./ (1 .+ abs.(V)))
        V = V_new
        iter += 1
    end

    if iter == max_iter
        println("recover_V!: reached max_iter with err = $err")
    else
        println("recover_V!: converged in $iter iterations (err = $err)")
    end

    res.V = V

    # calculate V on the histogram grid and the welfare measure W = E[V(a, s)] under μ
    V_hist = zeros(n_hist, ns)
    for s_idx in 1:ns
        V_hist[:, s_idx] = evaluate_spline(PchipSpline(a_grid, V[:, s_idx]), a_hist)
    end
    res.V_hist = V_hist
    res.W = res.μ ⋅ V_hist
end



# Keyword arguments override the Primitives defaults, e.g. solve_model(; a_min=-1.0, na=200)
solve_model(; kwargs...) = solve_model(Primitives(; kwargs...))

function solve_model(prim::Primitives)
    res = initialize(prim)
    steady_state_equilibrium!(prim, res)
    calculate_aggregates!(prim, res)
    recover_V!(prim, res; tol=1e-12)
    return prim, res
end
