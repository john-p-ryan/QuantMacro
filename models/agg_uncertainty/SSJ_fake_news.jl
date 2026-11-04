# This file computes sequence-space Jacobians for the Krusell-Smith model with the full "fake news" algorithm of
# Auclert, Bardóczy, Rognlie, & Straub (2021, Section 3.2), and uses them to solve for impulse responses.
#
# SSJ2.jl applies their Lemma 1: one backward iteration delivers the policies for a shock at every date s. It then still
# iterates the distribution forward separately for each s, which takes O(T²) sparse steps per input. Proposition 1
# removes those forward iterations. The Jacobian of an output with respect to an input is built from
#   𝒴ₛ  the effect of a shock at date s on the output at date 0,
#   𝒟ₛ  the effect of a shock at date s on the distribution at date 1,
#   ℰₜ  the expectation vectors (Λₛₛ)ᵗ yₛₛ: in the steady state, the expected outcome t periods ahead from each state,
# where Λₛₛ is the steady-state transition matrix (res.T_star') and yₛₛ the outcome on the histogram grid. 𝒴 and 𝒟 come
# from one backward iteration per input and ℰ from T steps of Λₛₛ per output, so the only O(T²) work left is a dense
# matrix product per input-output pair.
#
# Dates follow the paper and start at 0, when the shock is announced; index t+1 of an array holds date t. The inputs
# are the rental rate r and the wage w. The outputs are capital at the start of the period K, which is predetermined
# at date 0 (so the first row of its Jacobian is zero, as in SSJ2.jl), and consumption C.

using Parameters, LinearAlgebra, SparseArrays, Optim
using Spline # custom spline package

# EGM backward step, firm block, and market clearing solve, shared with SSJ2.jl (skipped if it is already loaded),
# which in turn loads the Aiyagari steady state
@isdefined(fast_jacobian) || include(joinpath(@__DIR__, "SSJ2.jl"))


# --- Steady-state transition on the histogram grid ---

# Young (2010) lottery, the rule build_T_star uses: a household that saves k′ moves to k_hist[i] with probability π
# and to k_hist[i+1] with probability 1 - π, where k_hist[i] ≤ k′ ≤ k_hist[i+1], so k′ is preserved in expectation.
# Savings off the grid go to its nearest end.
struct Lottery
    i::Matrix{Int}        # lower grid point, n_hist × nz
    π::Matrix{Float64}    # probability of the lower grid point
end

function Lottery(k_hist::AbstractVector, k′::AbstractMatrix)
    i = clamp.(searchsortedlast.(Ref(k_hist), k′), 1, length(k_hist) - 1)
    π = clamp.((k_hist[i .+ 1] .- k′) ./ (k_hist[i .+ 1] .- k_hist[i]), 0.0, 1.0)
    return Lottery(i, π)
end

# Lemma 3: ℰₜ = Λₛₛ ℰₜ₋₁. Today's expectation of tomorrow's outcome E, averaged first over tomorrow's employment
# state, then over the lottery.
function expectation_step(lottery::Lottery, M, E)
    EM = E * M'  # EM[k′, z] = Σ_z′ M[z, z′] E[k′, z′]
    E_today = similar(E)
    @inbounds for z in axes(E, 2), k in axes(E, 1)
        i, π = lottery.i[k, z], lottery.π[k, z]
        E_today[k, z] = π * EM[i, z] + (1 - π) * EM[i+1, z]
    end
    return E_today
end

# Expectation vectors ℰ₀ = y, ℰ₁, …, ℰₙ₋₁ of an outcome y on the histogram grid, one per column, flattened like vec(res.μ)
function expectation_vectors(lottery::Lottery, M, y, n)
    ℰ = zeros(length(y), n)
    ℰ[:, 1] = vec(y)
    E = y
    for t in 2:n
        E = expectation_step(lottery, M, E)
        ℰ[:, t] = vec(E)
    end
    return ℰ
end

# Change in tomorrow's distribution when savings move by dk′ from the steady state, given today's distribution D.
# Holding the lottery's grid points fixed, mass D dk′ / (k_hist[i+1] - k_hist[i]) moves from k_hist[i] to k_hist[i+1];
# employment then transitions with M.
function forward_shock(lottery::Lottery, k_hist, M, D, dk′)
    dD = zeros(size(D))
    @inbounds for z in axes(D, 2), k in axes(D, 1)
        i = lottery.i[k, z]
        shift = D[k, z] * dk′[k, z] / (k_hist[i+1] - k_hist[i])
        dD[i, z] -= shift
        dD[i+1, z] += shift
    end
    return dD * M
end


# --- Household policies on the histogram grid ---

# Savings policy on k_grid interpolated to the histogram grid, as in policy_iteration!, and kept on the grid as the
# lottery keeps it: a household whose policy exceeds k_max saves k_max. Consumption then absorbs the difference, so
# that savings and consumption add up to cash on hand and their responses are consistent with the distribution's.
function savings_on_hist(prim::Primitives, k_pol)
    k′ = zeros(prim.n_hist, prim.nz)
    for z in 1:prim.nz
        k′[:, z] = evaluate_spline(PchipSpline(prim.k_grid, k_pol[:, z]), prim.k_hist)
    end
    return clamp.(k′, first(prim.k_hist), last(prim.k_hist))
end

# Consumption on the histogram grid from the budget constraint c = (1 + r - δ)k + w ē ϵ - k′, so that budgets hold
# exactly on the grid the distribution lives on
consumption(prim::Primitives, r, w, k′) = (1 + r - prim.δ) .* prim.k_hist .+ w * prim.ē .* prim.z_grid' .- k′

# Lemma 1: households care only about how far ahead a price change is, so one backward iteration from a change at the
# last date T-1 delivers the date-0 policies for a change at every date s = 0, …, T-1. Returns savings and consumption
# on the histogram grid, n_hist × nz × T, where [:, :, s+1] holds the date-0 policies when prices change by (dr, dw)
# at date s.
function anticipation_policies(prim::Primitives, res::Results, T::Int; dr=0.0, dw=0.0)
    ∂V = (1 + res.r - prim.δ) .* u_prime.(res.c_policy, prim.γ)  # steady-state marginal value after the change
    k′ = zeros(prim.n_hist, prim.nz, T)
    c = zeros(prim.n_hist, prim.nz, T)
    for s in 0:T-1
        r = s == 0 ? res.r + dr : res.r
        w = s == 0 ? res.w + dw : res.w
        k_pol, _, ∂V = backward_iterate(prim, ∂V, r, w)
        k′[:, :, s+1] = savings_on_hist(prim, k_pol)
        c[:, :, s+1] = consumption(prim, r, w, k′[:, :, s+1])
    end
    return k′, c
end


# --- Fake news algorithm ---

# Step 1 for one input (:r or :w): 𝒴ₛ for consumption and 𝒟ₛ, per unit change in the input, one column per date s.
# Policies are differentiated by central differences, the two-sided variant in the paper's Appendix C.1. It needs no
# "ghost run": any drift of the backward iteration away from the steady state is common to both sides and cancels.
function input_shocks(prim::Primitives, res::Results, lottery::Lottery, T::Int, input::Symbol; h=1e-5)
    input in (:r, :w) || throw(ArgumentError("input must be :r or :w, got :$input"))
    change(x) = input === :r ? (dr=x, dw=0.0) : (dr=0.0, dw=x)
    k′_up, c_up = anticipation_policies(prim, res, T; change(h)...)
    k′_down, c_down = anticipation_policies(prim, res, T; change(-h)...)
    dk′ = (k′_up .- k′_down) ./ 2h
    dc = (c_up .- c_down) ./ 2h

    𝒴_C = [sum(view(dc, :, :, s) .* res.μ) for s in 1:T]
    𝒟 = zeros(length(res.μ), T)
    for s in 1:T
        𝒟[:, s] = vec(forward_shock(lottery, prim.k_hist, prim.M, res.μ, view(dk′, :, :, s)))
    end
    return 𝒴_C, 𝒟
end

# Step 3 (equation 26): row 1 of the fake news matrix holds 𝒴ₛ and row t+1 holds ℰₜ₋₁'𝒟ₛ, the effect at date t ≥ 1 of
# news about date s that arrives at date 0 and is retracted at date 1
fake_news_matrix(𝒴, ℰ, 𝒟) = [𝒴'; ℰ' * 𝒟]

# Step 4 (Proposition 1): J[t, s] = J[t-1, s-1] + F[t, s], so each entry of the Jacobian sums the fake news matrix
# along the diagonal that ends there
function jacobian_from_fake_news(F)
    J = copy(F)
    for s in 2:size(J, 2), t in 2:size(J, 1)
        J[t, s] += J[t-1, s-1]
    end
    return J
end

"""
    fake_news_jacobians(prim, res, T; h=1e-5)

Household Jacobians of capital at the start of the period (K) and consumption (C) with respect to the rental rate (r)
and the wage (w), truncated to T × T, from the fake news algorithm. `h` is the step for central differences of the
policies. Returns a NamedTuple whose entry `J.K.r[t+1, s+1]` = ∂K_t/∂r_s is the response of capital at date t to
news, at date 0, of a higher rental rate at date s. `J.K.r` and `J.K.w` follow the convention of `fast_jacobian` in
SSJ2.jl and can replace its output.
"""
function fake_news_jacobians(prim::Primitives, res::Results, T::Int; h=1e-5)
    @unpack k_hist, nz, M = prim
    k′_ss = savings_on_hist(prim, res.k_policy)  # res.k_pol_hist, kept on the grid
    lottery = Lottery(k_hist, k′_ss)

    # Step 2: expectation vectors of each output, which depend only on the steady state
    ℰ_K = expectation_vectors(lottery, M, repeat(k_hist, 1, nz), T - 1)
    ℰ_C = expectation_vectors(lottery, M, consumption(prim, res.r, res.w, k′_ss), T - 1)

    # Steps 1, 3, and 4 for each input. Capital at date 0 is predetermined, so its 𝒴 is zero.
    J = map((:r, :w)) do input
        𝒴_C, 𝒟 = input_shocks(prim, res, lottery, T, input; h)
        (K = jacobian_from_fake_news(fake_news_matrix(zeros(T), ℰ_K, 𝒟)),
         C = jacobian_from_fake_news(fake_news_matrix(𝒴_C, ℰ_C, 𝒟)))
    end
    return (K = (r = J[1].K, w = J[2].K), C = (r = J[1].C, w = J[2].C))
end


"""
    SolveFakeNews_IRFs(prim, res; T=300, h=1e-5, ρ=0.75, Z_shock=0.01)

General equilibrium responses to the TFP path dZ_t = Z_shock ρᵗ, using the fake news Jacobians. Capital solves the
linearized capital market clearing condition (`solve_dK` in SSJ2.jl); prices and consumption then follow from the
firm and household Jacobians. Returns the paths' deviations from the steady state, and the IRFs of capital and
consumption in percent deviations per percent shock.
"""
function SolveFakeNews_IRFs(prim::Primitives, res::Results; T=300, h=1e-5, ρ=0.75, Z_shock=0.01)  # ρ default shared with the KS, BKM, and Reiter solutions
    J = fake_news_jacobians(prim, res, T; h)
    J_r_K, J_w_K, J_r_Z, J_w_Z = Firm_Block(prim, res, T)
    dZ, dK = solve_dK(ρ, Z_shock, J.K.r, J.K.w, J_r_K, J_w_K, J_r_Z, J_w_Z)

    dr = J_r_K * dK + J_r_Z * dZ
    dw = J_w_K * dK + J_w_Z * dZ
    dC = J.C.r * dr + J.C.w * dw

    C_ss = sum(consumption(prim, res.r, res.w, savings_on_hist(prim, res.k_policy)) .* res.μ)  # steady state of the C output
    return (; dZ, dK, dr, dw, dC, irf_K = dK ./ (res.K * Z_shock), irf_C = dC ./ (C_ss * Z_shock))
end


# Main workflow
#=
prim_ss, res_ss = solve_model();
T = 300

# Same Jacobians as SSJ2.jl's direct method, without iterating the distribution forward for every shock date
@time J_K_r, J_K_w = fast_jacobian(prim_ss, res_ss, 1e-5, 1e-5, T);
@time J = fake_news_jacobians(prim_ss, res_ss, T);
maximum(abs.(J.K.r - J_K_r)) / maximum(abs.(J_K_r))

irfs = SolveFakeNews_IRFs(prim_ss, res_ss; T);

using Plots
plot(0:99, [irfs.irf_K[1:100] irfs.irf_C[1:100]],
     label=["Capital" "Consumption"],
     xlabel="Quarters after shock",
     ylabel="% deviation per % shock",
     title="Impulse Responses to a TFP Shock (ρ=0.75)",
     lw=2)
=#
