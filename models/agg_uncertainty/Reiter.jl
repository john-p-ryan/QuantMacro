# this file contains the functions to linearize Aiyagari model around the steady state
# using Reiter's method. The linearized system is then solved using the SGU method.

using ForwardDiff

# Function to create arrays that can store dual numbers if needed
function create_dual_compatible_array(T, dims...)
    return zeros(T, dims...)
end


"""
    construct_reiter_system(prim, res_ss; ρ=0.75, σ=0.00661)

Constructs the linearized system for Reiter's method around the steady state of the Aiyagari model.
Uses complementary slackness to properly handle binding borrowing constraints.

The system is redefined to avoid redundancies:
- State variables (x): [Z, μ (flattened, excluding the last element for each employment state)]
- Control variables (y): [k_policy (flattened)]
- Prices (r, w) and aggregate capital (K) are computed from states and controls

Returns a NamedTuple containing:
- Jacobians (J_x, J_y, J_x′, J_y′, J_ε)
- Steady state vectors (x_ss, y_ss)
- Dimensions (n_x, n_y)
- Function f and steady state vector v_ss
"""
function construct_reiter_system(prim, res_ss; ρ=0.75, σ=0.00661)
    # Extract relevant quantities from the steady state
    @unpack_Primitives prim
    @unpack_Results res_ss
    
    # Define dimensions for our state and control vectors
    # States: x = [Z, μ (flattened, excluding the last element for each employment state)]
    # We exclude n_ϵ elements from μ (one for each employment state)
    n_μ = n_hist * nϵ - nϵ  # Reduced by nϵ elements
    n_x = 1 + n_μ
    
    # Controls: y = [k_policy (flattened)]
    n_policy = nk * nϵ
    n_y = n_policy
    
    # Define the steady state vectors, excluding redundant distribution elements
    Z_ss = 1.0
    
    # Create a reduced μ vector (excluding last element for each employment state)
    μ_reduced = zeros(n_μ)
    idx = 1
    for ϵ_idx in 1:nϵ
        for k_idx in 1:(n_hist-1)  # Skip the last element for each ϵ
            μ_reduced[idx] = μ[k_idx, ϵ_idx]
            idx += 1
        end
    end
    
    x_ss = [Z_ss; μ_reduced]
    y_ss = vec(k_policy)
    
    # Combined vector at steady state
    v_ss = [x_ss; y_ss; x_ss; y_ss; 0.0]  # [x, y, x', y', ε]
    
    # Store indices for different components in the combined vector
    idxs = Dict(
        :x => 1:n_x,
        :y => (n_x+1):(n_x+n_y),
        :x′ => (n_x+n_y+1):(2*n_x+n_y),
        :y′ => (2*n_x+n_y+1):(2*n_x+2*n_y),
        :ε => length(v_ss)
    )

    # Function that evaluates equilibrium conditions
    function f(v)
        # Get element type for array creation
        T = eltype(v)
        
        # Extract components using standard indexing (not views)
        x_indices = idxs[:x]
        y_indices = idxs[:y]
        x′_indices = idxs[:x′]
        y′_indices = idxs[:y′]
        
        # Extract state variables
        Z = v[x_indices[1]]
        μ_flat_reduced = v[x_indices[2:end]]
        
        # Reshape μ back into a full matrix with the constraint that distributions sum to fixed values
        μ_t = create_dual_compatible_array(T, n_hist, nϵ)
        idx = 1
        for ϵ_idx in 1:nϵ
            # Set the first n_hist-1 elements for each ϵ
            for k_idx in 1:(n_hist-1)
                μ_t[k_idx, ϵ_idx] = μ_flat_reduced[idx]
                idx += 1
            end
            
            # For unemployment state (ϵ_idx = 2), ensure sum equals unemployment rate
            # For employment state (ϵ_idx = 1), ensure sum equals employment rate
            target_sum = (ϵ_idx == 2) ? unemp : (1 - unemp)
            
            # Set the last element such that the sum equals the target
            current_sum = sum(μ_t[1:(n_hist-1), ϵ_idx])
            μ_t[n_hist, ϵ_idx] = target_sum - current_sum
        end
        
        # Extract control variables (only capital policy)
        k_policy_flat = v[y_indices]
        
        # Reshape k_policy into a matrix safely
        k_policy_t = create_dual_compatible_array(T, nk, nϵ)
        for j in 1:nϵ
            for i in 1:nk
                idx = (j-1)*nk + i
                if idx <= length(k_policy_flat)
                    k_policy_t[i,j] = k_policy_flat[idx]
                end
            end
        end
        
        # Extract future state variables
        Z′ = v[x′_indices[1]]
        μ′_flat_reduced = v[x′_indices[2:end]]
        
        # Reshape μ′ back into a full matrix with constraints
        μ_t′ = create_dual_compatible_array(T, n_hist, nϵ)
        idx = 1
        for ϵ_idx in 1:nϵ
            # Set the first n_hist-1 elements for each ϵ
            for k_idx in 1:(n_hist-1)
                μ_t′[k_idx, ϵ_idx] = μ′_flat_reduced[idx]
                idx += 1
            end
            
            # Set the last element to maintain the sum constraint
            target_sum = (ϵ_idx == 2) ? unemp : (1 - unemp)
            current_sum = sum(μ_t′[1:(n_hist-1), ϵ_idx])
            μ_t′[n_hist, ϵ_idx] = target_sum - current_sum
        end
        
        # Extract future control variables (only capital policy)
        k_policy′_flat = v[y′_indices]
        
        # Reshape k_policy′ into a matrix safely
        k_policy_t′ = create_dual_compatible_array(T, nk, nϵ)
        for j in 1:nϵ
            for i in 1:nk
                idx = (j-1)*nk + i
                if idx <= length(k_policy′_flat)
                    k_policy_t′[i,j] = k_policy′_flat[idx]
                end
            end
        end
        
        # Extract shock
        ε = v[idxs[:ε]]
        
        # ---------- Compute derived variables ----------
        
        # Compute aggregate capital from distribution
        K_t = zero(T)
        for ϵ_index in 1:nϵ
            for k_index in 1:n_hist
                K_t += k_hist[k_index] * μ_t[k_index, ϵ_index]
            end
        end
        
        # Compute future aggregate capital
        K_t′ = zero(T)
        for ϵ_index in 1:nϵ
            for k_index in 1:n_hist
                K_t′ += k_hist[k_index] * μ_t′[k_index, ϵ_index]
            end
        end
        
        # Compute prices from aggregate variables
        r_t = α * Z * (L / K_t)^(1-α)
        w_t = (1-α) * Z * (K_t / L)^α
        r_t′ = α * Z′ * (L / K_t′)^(1-α)
        w_t′ = (1-α) * Z′ * (K_t′ / L)^α
        
        # ---------- Compute equilibrium conditions ----------
        
        # Derive consumption policy from budget constraint
        c_policy_t = create_dual_compatible_array(T, nk, nϵ)
        for ϵ_index in 1:nϵ
            ϵ_val = ϵ_grid[ϵ_index]
            for k_index in 1:nk
                k_val = k_grid[k_index]
                # Fix: Add ē to labor income
                c_policy_t[k_index, ϵ_index] = (1+r_t-δ) * k_val + w_t * ē * ϵ_val - k_policy_t[k_index, ϵ_index]
            end
        end
        
        # Derive future consumption policy
        c_policy_t′ = create_dual_compatible_array(T, nk, nϵ)
        for ϵ_index in 1:nϵ
            ϵ_val = ϵ_grid[ϵ_index]
            for k_index in 1:nk
                k_val = k_grid[k_index]
                # Fix: Add ē to labor income
                c_policy_t′[k_index, ϵ_index] = (1+r_t′-δ) * k_val + w_t′ * ē * ϵ_val - k_policy_t′[k_index, ϵ_index]
            end
        end
        
        # Compute interpolated policy on histogram grid
        k_pol_hist_t = create_dual_compatible_array(T, n_hist, nϵ)
        for ϵ_index in 1:nϵ
            for k_hist_index in 1:n_hist
                k_val = k_hist[k_hist_index]
                k_pol_hist_t[k_hist_index, ϵ_index] = safe_pchip(k_grid, k_policy_t[:, ϵ_index], k_val)
                
                # Enforce borrowing constraint on histogram grid
                if k_pol_hist_t[k_hist_index, ϵ_index] < k_min
                    k_pol_hist_t[k_hist_index, ϵ_index] = k_min
                end
            end
        end
        
        # Law of motion for the distribution
        μ_next = create_dual_compatible_array(T, n_hist, nϵ)
        for ϵ′_index in 1:nϵ
            for ϵ_index in 1:nϵ
                trans_prob = M[ϵ_index, ϵ′_index]
                for k_index in 1:n_hist
                    k′ = k_pol_hist_t[k_index, ϵ_index]
                    μ_mass = μ_t[k_index, ϵ_index]
                    
                    # Distribute to histogram bins using linear interpolation
                    for hist_idx in 1:n_hist-1
                        if k′ >= k_hist[hist_idx] && k′ <= k_hist[hist_idx+1]
                            weight_high = (k′ - k_hist[hist_idx]) / (k_hist[hist_idx+1] - k_hist[hist_idx])
                            weight_low = 1.0 - weight_high
                            
                            μ_next[hist_idx, ϵ′_index] += weight_low * trans_prob * μ_mass
                            μ_next[hist_idx+1, ϵ′_index] += weight_high * trans_prob * μ_mass
                            break
                        end
                    end
                    
                    # Handle edge cases
                    if k′ < k_hist[1]
                        μ_next[1, ϵ′_index] += trans_prob * μ_mass
                    elseif k′ > k_hist[end]
                        μ_next[end, ϵ′_index] += trans_prob * μ_mass
                    end
                end
            end
        end
        
        # TFP process
        log_Z′_expected = ρ * log(Z) + σ * ε
        
        # ---------- Improved handling of borrowing constraints and Euler equations ----------
        
        # Household optimization with smooth transition between regimes
        euler_errors = create_dual_compatible_array(T, nk, nϵ)
        for ϵ_index in 1:nϵ
            for k_index in 1:nk
                c_current = c_policy_t[k_index, ϵ_index]
                k_next = k_policy_t[k_index, ϵ_index]
                
                # Distance from borrowing constraint
                constraint_distance = k_next - k_min
                
                # Compute standard Euler equation
                # Expected marginal utility tomorrow
                expected_muc = zero(T)
                for ϵ′_index in 1:nϵ
                    # Interpolate consumption given k_next
                    c_next = safe_pchip(k_grid, c_policy_t′[:, ϵ′_index], k_next)
                    
                    # For log utility, MU = 1/c
                    muc_next = u_prime(c_next)
                    expected_muc += M[ϵ_index, ϵ′_index] * muc_next
                end
                
                # Standard Euler equation: MU_c(today) = β * (1+r′-δ) * E[MU_c(tomorrow)]
                muc_today = u_prime(c_current)
                euler_value = muc_today - β * (1+r_t′-δ) * expected_muc
                
                # Smooth transition between constraint and Euler equation
                # Small positive value to determine how close to constraint we need to be
                const_eps = 1e-4
                
                # Weight for constraint vs Euler equation
                # Weight is 1 when constraint binds strongly, 0 when far from constraint
                weight = max(0.0, min(1.0, 1.0 - constraint_distance / const_eps))
                
                # Weighted combination of constraint and Euler equation
                euler_errors[k_index, ϵ_index] = weight * constraint_distance + (1 - weight) * euler_value
            end
        end
        
        # Collect all equilibrium conditions in a vector
        # For the distribution, only include equations for the first n_hist-1 elements of each ϵ column
        # This avoids the redundant equations
        μ_diff_reduced = create_dual_compatible_array(T, n_hist-1, nϵ)
        for ϵ_index in 1:nϵ
            for k_index in 1:(n_hist-1)
                μ_diff_reduced[k_index, ϵ_index] = μ_t′[k_index, ϵ_index] - μ_next[k_index, ϵ_index]
            end
        end
        
        conditions = [
            vec(μ_diff_reduced);        # Distribution law of motion (reduced)
            log(Z′) - log_Z′_expected;  # TFP process
            vec(euler_errors);          # Euler equations (with complementary slackness)
        ]
        
        return conditions
    end
    
    # Compute Jacobians numerically with ForwardDiff
    println("Computing Jacobians...")
    # Using a smaller chunk size to avoid memory issues
    chunk_size = min(5, length(v_ss))
    jac_cfg = ForwardDiff.JacobianConfig(f, v_ss, ForwardDiff.Chunk{chunk_size}())
    
    # This can take a while for large systems
    J = ForwardDiff.jacobian(f, v_ss, jac_cfg)
    println("Jacobian computation complete.")
    
    # Extract the specific Jacobians
    J_x = J[:, idxs[:x]]
    J_y = J[:, idxs[:y]]
    J_x′ = J[:, idxs[:x′]]
    J_y′ = J[:, idxs[:y′]]
    J_ε = J[:, idxs[:ε]:idxs[:ε]]
    
    # Return all the relevant pieces in a named tuple
    return (
        J_x = J_x, 
        J_y = J_y, 
        J_x′ = J_x′, 
        J_y′ = J_y′, 
        J_ε = J_ε,
        x_ss = x_ss, 
        y_ss = y_ss, 
        n_x = n_x, 
        n_y = n_y, 
        f = f, 
        v_ss = v_ss,
        idxs = idxs
    )
end







"""
    verify_steady_state(system)

Verifies that the function f evaluates to approximately zero at the steady state.
"""
function verify_steady_state(system)
    f_ss = system.f(system.v_ss)
    max_error = maximum(abs.(f_ss))
    total_error = sum(abs.(f_ss))
    println("Maximum absolute error at steady state: ", max_error)
    println("Total absolute error at steady state: ", total_error)
    return max_error
end

"""
    describe_system(system)

Prints information about the dimensions and structure of the linearized system.
"""
function describe_system(system)
    @unpack n_x, n_y, J_x, J_y, J_x′, J_y′, J_ε = system
    
    n_eqs = size(J_x, 1)
    
    println("System dimensions:")
    println("  - State variables (x): $n_x")
    println("  - Control variables (y): $n_y")
    println("  - Total equations: $n_eqs")
    println()
    println("Jacobian dimensions:")
    println("  - J_x: $(size(J_x))")
    println("  - J_y: $(size(J_y))")
    println("  - J_x′: $(size(J_x′))")
    println("  - J_y′: $(size(J_y′))")
    println("  - J_ε: $(size(J_ε))")
    
    # Check if the system is well-formed
    if n_eqs == n_x + n_y
        println("\nThe system appears to be well-formed (number of equations equals number of unknowns).")
    else
        println("\nWARNING: The system has $n_eqs equations but $(n_x + n_y) unknowns.")
    end
end






"""
    solve_eig(A::Array{T,2}, B::Array{T,2}, n_x::Int) where T<: AbstractFloat

Returns the transition and policy functions following the method SGU (2004) delineates. `A` and `B` are defined as in SGU. `n_x` is the number of states.
"""
function solve_eig(A::Array{T,2}, B::Array{T,2}, n_x::Int) where T<: AbstractFloat

    F = eigen(B,A)
    perm = sortperm(abs.(F.values))
    V = F.vectors[:,perm]
    D = F.values[perm]
    m = findlast(abs.(D) .< 1)
    eu = [true,true]

    if m > n_x
        eu[2] = false
        println("WARNING: the equilibrium is not unique !")
    elseif m < n_x
        eu[1] = false
        println("WARNING: the equilibrium does not exist !")
    end

    if all(eu)
        h_x = V[1:m,1:m]*Diagonal(D)[1:m,1:m]*inv(V[1:m,1:m])
        g_x = V[m+1:end,1:m]*inv(V[1:m,1:m])
        return (real.(g_x), real.(h_x), eu)
    end

end





"""
    generate_tfp_shock_response(system, g_x, h_x; shock_size=0.01, T=40)

Compute impulse response function (IRF) for a TFP shock.

Parameters:
- system: System returned by construct_reiter_system
- g_x: Policy function matrix from solve_eig
- h_x: State transition matrix from solve_eig
- shock_size: Size of the TFP shock (typically 0.01 for a 1% shock)
- T: Number of periods to compute IRF for

Returns:
- irf_x: State variable responses
- irf_y: Control variable responses
"""
function generate_tfp_shock_response(system, g_x, h_x; shock_size=0.01, T=40)
    @unpack n_x, n_y, x_ss, y_ss = system
    
    # Initialize IRF matrices for deviations from steady state
    irf_x = zeros(n_x, T+1)
    irf_y = zeros(n_y, T+1)
    
    # Set initial shock to TFP (first state variable)
    # Since TFP process is in logs, apply shock to log(Z)
    irf_x[1, 1] = shock_size
    
    # Compute control variables for the initial period
    irf_y[:, 1] = g_x * irf_x[:, 1]
    
    # Propagate state and control variables
    for t in 1:T
        irf_x[:, t+1] = h_x * irf_x[:, t]
        irf_y[:, t+1] = g_x * irf_x[:, t+1]
    end
    
    return irf_x, irf_y
end

"""
    compute_aggregate_capital(prim, x, μ_flat_reduced)

Compute aggregate capital from the reduced distribution.

Parameters:
- prim: Primitives structure for the model
- x: State vector [Z, μ (flattened, excluding the last element for each employment state)]

Returns:
- K: Aggregate capital
"""
function compute_aggregate_capital(prim, x)
    @unpack k_hist, n_hist, nϵ, unemp = prim
    
    Z = x[1]
    μ_flat_reduced = x[2:end]
    
    μ = zeros(n_hist, nϵ)
    idx = 1
    for ϵ_idx in 1:nϵ
        for k_idx in 1:(n_hist-1)
            μ[k_idx, ϵ_idx] = μ_flat_reduced[idx] 
            idx += 1
        end
        target_sum = (ϵ_idx == 2) ? unemp : (1 - unemp)
        current_sum = sum(μ[1:(n_hist-1), ϵ_idx])
        μ[n_hist, ϵ_idx] = target_sum - current_sum  # No rescaling!
    end
    
    # Compute aggregate capital
    K = sum(k_hist[k_idx] * μ[k_idx, ϵ_idx] for ϵ_idx in 1:nϵ for k_idx in 1:n_hist)
    return K
end



# Calculate IRFs using Reiter's method
function calculate_reiter_irfs(prim, res_ss, system, g_x, h_x; shock_size=0.01, T=40)
    # Generate the basic IRFs for state and control variables
    irf_x, irf_y = generate_tfp_shock_response(system, g_x, h_x; shock_size=shock_size, T=T)
    
    # Extract primitives and steady state values
    @unpack α, δ, L = prim
    @unpack K, r, w = res_ss
    
    # Initialize IRFs for variables of interest (percentage deviations)
    Z_irf = zeros(T+1)
    K_irf = zeros(T+1)
    r_irf = zeros(T+1)
    w_irf = zeros(T+1)
    C_irf = zeros(T+1)
    K_var_irf = zeros(T+1)
    
    # Calculate steady state consumption and capital variance
    Y_ss = K^α * L^(1-α)
    I_ss = δ * K
    C_ss = Y_ss - I_ss
    
    # Calculate IRFs for each period
    for t in 1:T+1
        # Extract TFP deviation (first state variable)
        Z_dev = irf_x[1, t]
        # Ensure TFP is positive to avoid domain errors
        Z_t = max(1e-6, 1.0 + Z_dev)
        Z_irf[t] = 100 * Z_dev  # Percentage deviation
        
        # Calculate aggregate capital from state vector
        x_t = system.x_ss + irf_x[:, t]
        K_t = compute_aggregate_capital(prim, x_t)
        K_irf[t] = 100 * (K_t - K) / K
        
        # Calculate prices using production function with safety checks
        r_t = α * Z_t * (L / K_t)^(1-α)
        w_t = (1-α) * Z_t * (K_t / L)^α
        
        r_irf[t] = 100 * (r_t - r) / r
        w_irf[t] = 100 * (w_t - w) / w
        
        # Calculate consumption using resource constraint
        Y_t = Z_t * K_t^α * L^(1-α)
        
        # For correct investment, we need K_{t+1}
        K_t_plus_1 = 0.0
        if t < T+1
            # Get state in next period
            x_t_plus_1 = system.x_ss + irf_x[:, t+1]
            K_t_plus_1 = compute_aggregate_capital(prim, x_t_plus_1)
        else
            # For the last period, use transition matrix to project state one more period
            x_t_plus_1_dev = h_x * irf_x[:, t]
            x_t_plus_1 = system.x_ss + x_t_plus_1_dev
            K_t_plus_1 = compute_aggregate_capital(prim, x_t_plus_1)
        end
        
        # Correct investment formula: I_t = K_{t+1} - (1-δ) * K_t
        I_t = K_t_plus_1 - (1-δ) * K_t
        C_t = Y_t - I_t
        
        C_irf[t] = 100 * (C_t - C_ss) / C_ss
        
        # For capital variance, we need to compute the distribution's second moment
        # Reconstruct distribution from state vector
        μ_flat_reduced = x_t[2:end]
        μ_t = zeros(prim.n_hist, prim.nϵ)
        idx = 1
        for ϵ_idx in 1:prim.nϵ
            # Set the first n_hist-1 elements for each ϵ
            for k_idx in 1:(prim.n_hist-1)
                μ_t[k_idx, ϵ_idx] = μ_flat_reduced[idx]
                idx += 1
            end
            
            # Set the last element to maintain sum constraints
            target_sum = (ϵ_idx == 2) ? prim.unemp : (1 - prim.unemp)
            current_sum = sum(μ_t[1:(prim.n_hist-1), ϵ_idx])
            μ_t[prim.n_hist, ϵ_idx] = target_sum - current_sum
        end
        
        # Calculate capital variance
        K_var_t = 0.0
        for ϵ_idx in 1:prim.nϵ
            for k_idx in 1:prim.n_hist
                K_var_t += μ_t[k_idx, ϵ_idx] * (prim.k_hist[k_idx] - K_t)^2
            end
        end
        
        # If we have steady state K_var from res_ss
        if hasproperty(res_ss, :K_var)
            K_var_irf[t] = 100 * (K_var_t - res_ss.K_var) / res_ss.K_var
        else
            # Calculate steady state K_var
            K_var_ss = 0.0
            for ϵ_idx in 1:prim.nϵ
                for k_idx in 1:prim.n_hist
                    K_var_ss += res_ss.μ[k_idx, ϵ_idx] * (prim.k_hist[k_idx] - K)^2
                end
            end
            K_var_irf[t] = 100 * (K_var_t - K_var_ss) / K_var_ss
        end
    end
    
    return (Z=Z_irf, K=K_irf, r=r_irf, w=w_irf, C=C_irf, K_var=K_var_irf)
end


function compute_capital_variance(prim, x, K)
    @unpack k_hist, n_hist, nϵ, unemp = prim
    
    # Extract and reconstruct distribution
    μ_flat_reduced = x[2:end]
    μ = zeros(n_hist, nϵ)
    idx = 1
    for ϵ_idx in 1:nϵ
        for k_idx in 1:(n_hist-1)
            μ[k_idx, ϵ_idx] = max(0.0, μ_flat_reduced[idx])
            idx += 1
        end
        target_sum = (ϵ_idx == 2) ? unemp : (1 - unemp)
        current_sum = sum(μ[1:(n_hist-1), ϵ_idx])
        if current_sum > target_sum
            scale_factor = target_sum / current_sum
            for k_idx in 1:(n_hist-1)
                μ[k_idx, ϵ_idx] *= scale_factor
            end
            μ[n_hist, ϵ_idx] = 0.0
        else
            μ[n_hist, ϵ_idx] = target_sum - current_sum
        end
    end
    
    # Compute variance
    K_var = 0.0
    for ϵ_idx in 1:nϵ
        for k_idx in 1:n_hist
            K_var += μ[k_idx, ϵ_idx] * (k_hist[k_idx] - K)^2
        end
    end
    
    return K_var
end



"""
    simulate_reiter_economy(Z_path, system, g_x, h_x, res_ss, prim; ρ=0.75)

Simulate the economy using Reiter's linearized state-space system.

Parameters:
- Z_path: Vector of TFP levels over time
- system: System structure from construct_reiter_system
- g_x: Policy function matrix from solve_eig
- h_x: State transition matrix from solve_eig
- res_ss: Steady state results
- prim: Primitives
- ρ: TFP persistence parameter

Returns:
- NamedTuple with simulated paths for K, K_var, C, r, w
"""
function simulate_reiter_economy(Z_path, system, g_x, h_x, res_ss, prim; ρ=0.75)
    @unpack x_ss, y_ss, n_x, n_y = system
    @unpack α, δ, L = prim
    
    T_sim = length(Z_path)
    
    # Back out log TFP innovations from the path
    log_Z_path = log.(Z_path)
    innovations = zeros(T_sim)
    innovations[1] = log_Z_path[1]  # Initial deviation from log(1)=0
    for t in 2:T_sim
        innovations[t] = log_Z_path[t] - ρ * log_Z_path[t-1]
    end
    
    # Initialize state and control deviations
    x_dev = zeros(n_x, T_sim)
    y_dev = zeros(n_y, T_sim)
    
    # Initialize output arrays
    K_path = zeros(T_sim)
    K_var_path = zeros(T_sim)
    C_path = zeros(T_sim)
    r_path = zeros(T_sim)
    w_path = zeros(T_sim)
    
    # Simulate forward
    for t in 1:T_sim
        # Apply TFP shock to first state variable
        if t == 1
            x_dev[1, t] = innovations[t]
        else
            # State transition: x_t = h_x * x_{t-1} + shock
            x_dev[:, t] = h_x * x_dev[:, t-1]
            x_dev[1, t] += innovations[t]  # Add TFP innovation
        end
        
        # Compute controls
        y_dev[:, t] = g_x * x_dev[:, t]
        
        # Current state and control in levels
        x_t = x_ss + x_dev[:, t]
        y_t = y_ss + y_dev[:, t]
        
        # Extract TFP (state variable is in LEVELS, not logs!)
        #Z_t = x_t[1]  # FIX: TFP is stored in levels in the state vector
        log_Z_dev = x_dev[1, t]
        Z_t = exp(log_Z_dev) 
        
        # Compute aggregate capital
        K_t = compute_aggregate_capital(prim, x_t)
        K_path[t] = K_t
        
        # Compute cross-sectional variance
        K_var_path[t] = compute_capital_variance(prim, x_t, K_t)
        
        # Compute prices
        r_t = α * Z_t * (L / K_t)^(1-α)
        w_t = (1-α) * Z_t * (K_t / L)^α
        r_path[t] = r_t
        w_path[t] = w_t
        
        # Compute output
        Y_t = Z_t * K_t^α * L^(1-α)
        
        # Compute investment (need next period's capital)
        if t < T_sim
            # Project next state
            x_t_plus_1_dev = h_x * x_dev[:, t]
            # For next period's innovation, need to check if we have it
            if t+1 <= T_sim
                x_t_plus_1_dev[1] += innovations[t+1]
            end
            x_t_plus_1 = x_ss + x_t_plus_1_dev
            K_t_plus_1 = compute_aggregate_capital(prim, x_t_plus_1)
        else
            # For last period, project one more step using transition matrix only
            x_t_plus_1_dev = h_x * x_dev[:, t]
            x_t_plus_1 = x_ss + x_t_plus_1_dev
            K_t_plus_1 = compute_aggregate_capital(prim, x_t_plus_1)
        end
        
        # Investment: I_t = K_{t+1} - (1-δ) * K_t
        I_t = K_t_plus_1 - (1-δ) * K_t
        
        # Consumption from resource constraint
        C_t = Y_t - I_t
        C_path[t] = C_t
    end
    
    return (K=K_path, K_var=K_var_path, C=C_path, r=r_path, w=w_path)
end