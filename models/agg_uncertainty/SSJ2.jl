# This file contains an algorithm for computing Sequence Space Jacobians
# for the Krusell-Smith model, based on Auclert, Bardóczy, Rognlie, & Straub (2021).

using Parameters, LinearAlgebra, SparseArrays

# Ensure the necessary functions and types from your original files are available
include("spline.jl")
include("Aiyagari_EGM.jl")

# We need the backward_iterate function from your original implementation.
function backward_iterate(prim_ss::Primitives, ∂V_next, r, w)
    @unpack_Primitives prim_ss 

    k_pol = zeros(nk, nϵ)
    c_pol = zeros(nk, nϵ)

    for (ϵ_index, ϵ) in enumerate(ϵ_grid)
        p = M[ϵ_index,:]
        EMU_prime = ∂V_next * p

        c_today = u_prime_inv.(β * EMU_prime)                                    
        k_today = (c_today + k_grid .- w * ē * ϵ) / (1+r-δ)                     
        c_spline = PchipSpline(k_today, c_today)                            
        c_pol[:, ϵ_index] = evaluate_spline(c_spline, k_grid)                
        k_pol[:, ϵ_index] = (1+r-δ) * k_grid .+ w * ē * ϵ - c_pol[:, ϵ_index]   

        binding = (k_pol[:, ϵ_index] .< k_min)
        if any(binding)
            k_pol[binding, ϵ_index] .= k_min
            c_pol[binding, ϵ_index] .= (1+r-δ) * k_grid[binding] .+ w * ē * ϵ .- k_min
        end
    end

    ∂V = (1+r-δ) * u_prime.(c_pol)

    return k_pol, c_pol, ∂V
end


"""
    compute_policy_sequences(prim_ss, res_ss, T; dr=0.0, dw=0.0)

Performs a single backward recursion to compute all relevant policy transition matrices
for a future one-time shock (`dr` or `dw`).

### Returns
- `T_shock`: The transition matrix for the policy *at the time of* the shock.
- `T_seq`: A `Vector` of `T` transition matrices. `T_seq[j]` is for the policy
           `j` periods *before* the shock.
"""
function compute_policy_sequences(prim_ss::Primitives, res_ss::Results, T::Int; dr::Float64=0.0, dw::Float64=0.0)
    @unpack_Primitives prim_ss
    res_temp = deepcopy(res_ss)

    # --- Step 1: Find policy and co-state AT the shock time `s` ---
    ∂V_ss = (1 + res_ss.r - δ) * u_prime.(res_ss.c_policy)
    r_shock, w_shock = res_ss.r + dr, res_ss.w + dw
    
    k_pol_shock, _, ∂V_at_shock = backward_iterate(prim_ss, ∂V_ss, r_shock, w_shock)

    # Build the transition matrix for the policy AT the shock
    for ϵ_index in 1:nϵ
        k_spline = PchipSpline(prim_ss.k_grid, k_pol_shock[:, ϵ_index])
        res_temp.k_pol_hist[:, ϵ_index] = evaluate_spline(k_spline, prim_ss.k_hist)
    end
    T_shock = BuildTransitionMatrix(prim_ss, res_temp)
    
    # --- Step 2: Perform backward recursion for policies BEFORE the shock ---
    T_seq = Vector{SparseMatrixCSC{Float64, Int}}(undef, T)
    ∂V_prev = ∂V_at_shock 

    for j in 1:T # j = periods before shock
        k_pol_j, _, ∂V_curr = backward_iterate(prim_ss, ∂V_prev, res_ss.r, res_ss.w)

        for ϵ_index in 1:nϵ
            k_spline = PchipSpline(prim_ss.k_grid, k_pol_j[:, ϵ_index])
            res_temp.k_pol_hist[:, ϵ_index] = evaluate_spline(k_spline, prim_ss.k_hist)
        end
        T_seq[j] = BuildTransitionMatrix(prim_ss, res_temp)
        ∂V_prev = ∂V_curr
    end

    return T_shock, T_seq
end


"""
    fast_jacobian(prim_ss, res_ss, dr, dw, T)

Computes the household block Jacobians, J_K_r and J_K_w, using the efficient two-stage method.
"""
function fast_jacobian(prim_ss::Primitives, res_ss::Results, dr::Float64, dw::Float64, T::Int)
    # --- Stage 1: Pre-computation ---
    println("Pre-computing transition matrix sequences for dr shock...")
    T_r_shock, T_r_seq = compute_policy_sequences(prim_ss, res_ss, T; dr=dr)
    println("Pre-computing transition matrix sequences for dw shock...")
    T_w_shock, T_w_seq = compute_policy_sequences(prim_ss, res_ss, T; dw=dw)
    println("Pre-computation complete.")

    # --- Stage 2: Jacobian Construction ---
    J_K_r, J_K_w = zeros(T, T), zeros(T, T)
    T_ss = res_ss.T_star
    μ_ss_vec = vec(res_ss.μ)
    k_hist_rep_vec = vec(repeat(prim_ss.k_hist, 1, prim_ss.nϵ))

    println("Constructing Jacobians...")
    # Loop over shock time `s` (columns of Jacobians)
    for s in 1:T
        μ_r_vec, μ_w_vec = μ_ss_vec, μ_ss_vec

        # Loop over response time `t` (rows of Jacobians)
        for t in 1:T
            # The distribution at `t` (μ_t) is determined by policy at `t-1`.
            # Let policy_time = t - 1.
            policy_time = t - 1

            # --- Update distribution for the `r` shock ---
            local T_r_t
            if policy_time < s
                T_r_t = T_r_seq[s - policy_time]
            elseif policy_time == s
                T_r_t = T_r_shock
            else # policy_time > s
                T_r_t = T_ss
            end
            μ_r_vec = T_r_t * μ_r_vec
            K_r_t = k_hist_rep_vec' * μ_r_vec
            J_K_r[t, s] = (K_r_t - res_ss.K) / dr
            
            # --- Update distribution for the `w` shock ---
            local T_w_t
            if policy_time < s
                T_w_t = T_w_seq[s - policy_time]
            elseif policy_time == s
                T_w_t = T_w_shock
            else # policy_time > s
                T_w_t = T_ss
            end
            μ_w_vec = T_w_t * μ_w_vec
            K_w_t = k_hist_rep_vec' * μ_w_vec
            J_K_w[t, s] = (K_w_t - res_ss.K) / dw
        end
    end
    println("Jacobian construction complete.")

    return J_K_r, J_K_w
end

function Firm_Block(prim_ss::Primitives, res_ss::Results, T::Int)
    @unpack_Primitives prim_ss
    @unpack_Results res_ss
    Z = 1.0 ### Note we are making this assumption at steady state

    # Calculate the static Jacobians for the firm block in closed form
    # we have r = α * Z * K^α * L^(1-α)
    # and w = (1-α) * Z * K^α * L^(-α)
    dr_dK = α * (α - 1) * Z * K^(α - 2) * L^(1 - α)
    dw_dK = α * (1-α) * Z * K^(α - 1) * L^(-α)
    dr_dZ = α * K^(α-1) * L^(1 - α)
    dw_dZ = (1-α) * K^α * L^(-α)

    J_r_K = diagm(fill(dr_dK, T))
    J_w_K = diagm(fill(dw_dK, T))
    J_r_Z = diagm(fill(dr_dZ, T))
    J_w_Z = diagm(fill(dw_dZ, T))
    
    return J_r_K, J_w_K, J_r_Z, J_w_Z
end


function solve_dK(ρ, Z_shock, J_K_r, J_K_w, J_r_K, J_w_K, J_r_Z, J_w_Z)
    T = size(J_K_r, 1)

    Z_path = ones(T)
    Z_path[1] += Z_shock
    for t in 2:T
        Z_path[t] = exp(ρ * log(Z_path[t-1]))
    end
    dZ = Z_path - ones(T)

    # Market clearing Jacobians using Chain Rule
    H_K = J_K_r * J_r_K + J_K_w * J_w_K - I
    H_Z = J_K_r * J_r_Z + J_K_w * J_w_Z

    # solve dK using Implicit Function Theorem since H ≡ 0
    dK = -H_K \ H_Z * dZ
    return dZ, dK
end

function SolveSSJ_IRFs(prim_ss::Primitives, res_ss::Results; T=300, dx=1e-5, ρ=0.75, Z_shock=0.01)
    # Step 1: Compute Household Jacobians (H_K)
    J_K_r, J_K_w = fast_jacobian(prim_ss, res_ss, dx, dx, T)

    # Step 2: Compute Firm Jacobians (H_Z)
    J_r_K, J_w_K, J_r_Z, J_w_Z = Firm_Block(prim_ss, res_ss, T)

    # Step 3: Solve for the IRF using the Implicit Function Theorem
    dZ, dK = solve_dK(ρ, Z_shock, J_K_r, J_K_w, J_r_K, J_w_K, J_r_Z, J_w_Z)

    # Step 4: Calculate percentage change in capital
    scale = 1 / Z_shock # scale factor for percentage change
    dK_dZ = scale * dK / res_ss.K # percentage change

    return dZ, dK, K_pct_change
end


# Main workflow
#=
println("Solving for the model's steady state...")
@time prim_ss, res_ss = SolveModel();
println("Steady state found. K_ss = $(res_ss.K), r_ss = $(res_ss.r)")

T = 300
dx = 1e-5 # Use a smaller perturbation for better accuracy

# --- Step 1: Compute Household Jacobians (H_K) ---
# This now calls the fast, efficient function
@time J_K_r, J_K_w = fast_jacobian(prim_ss, res_ss, dx, dx, T);

# --- Step 2: Compute Firm Jacobians (H_Z) ---
# This part remains the same
J_r_K, J_w_K, J_r_Z, J_w_Z = Firm_Block(prim_ss, res_ss, T);

# --- Step 3: Solve for the IRF using the Implicit Function Theorem ---
ρ = 0.75
Z_shock = 0.01
@time dZ, dK = solve_dK(ρ, Z_shock, J_K_r, J_K_w, J_r_K, J_w_K, J_r_Z, J_w_Z);


# --- Step 4: Plotting Results ---
K_pct_change = dK / res_ss.K * 100 # percentage change

# The IRF of aggregate capital to a 1% TFP shock
using Plots
plot(1:100, K_pct_change[1:100],
     label="Capital IRF (% dev. from SS)",
     xlabel="Quarters after shock",
     ylabel="% deviation",
     title="Impulse Response to a 1% TFP Shock (ρ=$ρ)",
     lw=2)
=#