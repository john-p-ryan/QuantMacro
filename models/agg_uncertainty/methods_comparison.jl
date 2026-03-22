# ==============================================================================
# COMPARATIVE ANALYSIS OF HETEROGENEOUS AGENT MODEL SOLUTION METHODS
# ==============================================================================
# This script performs a comparative analysis of four methods for solving and
# simulating the Krusell-Smith model:
# 1. Krusell-Smith (KS): The original non-linear solution with EGM and histograms.
# 2. Boppart-Krusell-Mitman (BKM): Linearization via a perfect foresight transition path.
# 3. Reiter's Method: A linear state-space approximation method.
# 4. Sequence-Space Jacobians (SSJ): A fast method utilizing "block Jacobians" in the sequence space.
#
# The script is organized as follows:
# - Part 0: Setup and Module Imports
# - Part 1: Baseline Analysis (1% TFP Shock)
#   - Times and solves each method.
#   - Compares the impulse responses for aggregate capital.
#   - Plots a panel of IRFs for the BKM method.
# - Part 2: Long Simulation and Moment Comparison (1% TFP Shock)
#   - Simulates a 10,000-period economy with KS and BKM.
#   - Compares key statistical moments.
# - Part 3: Analysis with Scaled-Up Shocks
#   - Repeats the IRF and moment comparisons for larger shocks (5%, 10%, 25%).
#   - This tests the accuracy of the linearization methods against the
#     non-linear benchmark when non-linearities become more important.
# ==============================================================================



using Plots, Parameters, Statistics, BenchmarkTools, Printf, Random, ForwardDiff
include("spline.jl")
include("Aiyagari_EGM.jl")

# Load modules for each method
module KS
    # Note: KS_EGM_hist.jl internally includes spline.jl and Aiyagari_EGM.jl
    include("KS_EGM_hist.jl")
end
include("BKM_transition.jl")
include("Reiter.jl")
include("SSJ2.jl")


# --- Helper function for printing moment comparison tables ---
function print_moments_table(moments_ks, moments_bkm, moments_reiter, shock_size)
    println("\n" * "="^80)
    @printf("Moment Comparison for %.0f%% TFP Shock\n", shock_size * 100)
    println("="^80)
    @printf("%-20s | %12s | %12s | %12s | %12s | %12s\n", 
            "Moment", "KS (Full)", "BKM (Seq)", "Reiter (State)", "BKM Diff", "Reiter Diff")
    println("-"^80)

    ks_K_mean = mean(moments_ks.K)
    bkm_K_mean = mean(moments_bkm.K)
    reiter_K_mean = mean(moments_reiter.K)
    ks_K_var = var(moments_ks.K)
    bkm_K_var = var(moments_bkm.K)
    reiter_K_var = var(moments_reiter.K)
    ks_K_var_mean = mean(moments_ks.K_var)
    bkm_K_var_mean = mean(moments_bkm.K_var)
    reiter_K_var_mean = mean(moments_reiter.K_var)
    ks_C_mean = mean(moments_ks.C)
    bkm_C_mean = mean(moments_bkm.C)
    reiter_C_mean = mean(moments_reiter.C)
    ks_r_mean = mean(moments_ks.r)
    bkm_r_mean = mean(moments_bkm.r)
    reiter_r_mean = mean(moments_reiter.r)
    ks_w_mean = mean(moments_ks.w)
    bkm_w_mean = mean(moments_bkm.w)
    reiter_w_mean = mean(moments_reiter.w)

    @printf("%-20s | %12.4f | %12.4f | %12.4f | %12.4f | %12.4f\n", 
            "Mean K", ks_K_mean, bkm_K_mean, reiter_K_mean, 
            abs(ks_K_mean - bkm_K_mean), abs(ks_K_mean - reiter_K_mean))
    @printf("%-20s | %12.4f | %12.4f | %12.4f | %12.4f | %12.4f\n", 
            "Var(K)", ks_K_var, bkm_K_var, reiter_K_var, 
            abs(ks_K_var - bkm_K_var), abs(ks_K_var - reiter_K_var))
    @printf("%-20s | %12.4f | %12.4f | %12.4f | %12.4f | %12.4f\n", 
            "Mean X-Sect Var(K)", ks_K_var_mean, bkm_K_var_mean, reiter_K_var_mean, 
            abs(ks_K_var_mean - bkm_K_var_mean), abs(ks_K_var_mean - reiter_K_var_mean))
    @printf("%-20s | %12.4f | %12.4f | %12.4f | %12.4f | %12.4f\n", 
            "Mean C", ks_C_mean, bkm_C_mean, reiter_C_mean, 
            abs(ks_C_mean - bkm_C_mean), abs(ks_C_mean - reiter_C_mean))
    @printf("%-20s | %12.4f | %12.4f | %12.4f | %12.4f | %12.4f\n", 
            "Mean r", ks_r_mean, bkm_r_mean, reiter_r_mean, 
            abs(ks_r_mean - bkm_r_mean), abs(ks_r_mean - reiter_r_mean))
    @printf("%-20s | %12.4f | %12.4f | %12.4f | %12.4f | %12.4f\n", 
            "Mean w", ks_w_mean, bkm_w_mean, reiter_w_mean, 
            abs(ks_w_mean - bkm_w_mean), abs(ks_w_mean - reiter_w_mean))
    println("="^80 * "\n")
end


function main()
    # --- Global Simulation Parameters ---
    T_irf = 500       # Horizon for IRFs
    T_plot = 250      # Horizon to show in plots
    T_sim = 10000     # Horizon for long simulation
    shock_sizes = [0.01, 0.05, 0.10, 0.25] # Shock sizes to test

    use_level_deviations = true # flag for BKM to use level or pct change

    # Store computation times
    timings = Dict()

    println("\n--- Part 1: Baseline Analysis (1% TFP Shock) ---")
    shock = shock_sizes[1]

    # --- 1a. Solve Steady State (common for all methods) ---
    println("\nSolving for the steady state...")
    # Use Aiyagari module from BKM as it's at the top level
    prim_ss, res_ss = SolveModel(; k_min=1e-6, k_max=60.0, nk=60, n_hist=125)
    println("Steady state solved. K_ss = $(res_ss.K)")

    # --- 1b. Solve and Time Each Method ---
    println("\nTiming and solving each method for a $(shock*100)% shock...")

    # Krusell-Smith
    println("\nSolving with Krusell-Smith...")
    timed_ks = @timed KS.SolveModel(; z_grid=[1.0+shock, 1.0-shock]);
    prim_ks, sim_ks, res_ks = timed_ks.value
    t_ks = timed_ks.time
    irf_ks_dev, _ = KS.estimate_impulse_response(prim_ks, res_ks, sim_ks, num_simulations=2_000_000, max_horizon=T_irf)
    K_bar_ks = mean(res_ks.K_sim_path[(sim_ks.burn+1):end])
    irf_ks = vcat(0.0, (irf_ks_dev ./ K_bar_ks) .* (1 / shock))[1:T_irf+1] # Normalized IRF
    timings["KS"] = t_ks
    

    # BKM
    println("\nSolving with BKM...")
    timed_bkm = @timed SolveTransition_from_SS(prim_ss, res_ss; T=T_irf, Z_shock=shock);
    sim_bkm, res_tr_bkm = timed_bkm.value
    t_bkm = timed_bkm.time
    irfs_bkm = ComputeIRFs(prim_ss, res_ss, res_tr_bkm, sim_bkm);
    timings["BKM"] = t_bkm

    # Reiter
    println("\nSolving with Reiter...")
    timed_reiter = @timed begin
        system = construct_reiter_system(prim_ss, res_ss)
        AA = hcat(system.J_x′, system.J_y′)
        BB = -hcat(system.J_x, system.J_y)
        g_x, h_x = solve_eig(AA, BB, system.n_x)
        calculate_reiter_irfs(prim_ss, res_ss, system, g_x, h_x; shock_size=shock, T=T_irf)
    end
    irfs_reiter = timed_reiter.value
    t_reiter = timed_reiter.time
    timings["Reiter"] = t_reiter
    
    # Store system and matrices for later simulation
    system_reiter = construct_reiter_system(prim_ss, res_ss)
    AA_reiter = hcat(system_reiter.J_x′, system_reiter.J_y′)
    BB_reiter = -hcat(system_reiter.J_x, system_reiter.J_y)
    g_x_reiter, h_x_reiter = solve_eig(AA_reiter, BB_reiter, system_reiter.n_x)

    # SSJ
    println("\nSolving with SSJ...")
    timed_ssj = @timed begin
        J_K_r, J_K_w = fast_jacobian(prim_ss, res_ss, 1e-5, 1e-5, T_irf)
        J_r_K, J_w_K, J_r_Z, J_w_Z = Firm_Block(prim_ss, res_ss, T_irf)
        solve_dK(0.75, shock, J_K_r, J_K_w, J_r_K, J_w_K, J_r_Z, J_w_Z)
    end
    _, dK_ssj = timed_ssj.value
    t_ssj = timed_ssj.time
    irf_ssj_K = (dK_ssj / res_ss.K) * (1 / shock) # Normalized IRF
    timings["SSJ"] = t_ssj

    # --- 1c. Print Computation Times ---
    println("\n" * "="^40)
    println("Computation Times (Baseline 1% Shock)")
    println("="^40)
    for (method, time) in timings
        @printf("%-10s: %.4f seconds\n", method, time)
    end
    println("="^40)

    # --- 1d. Compare IRFs for Aggregate Capital ---
    println("\nPlotting IRF comparison for K...")
    t_range = 0:T_plot
    plot_irf_comp = plot(t_range, irfs_bkm.K[1:T_plot+1], label="BKM", lw=2,
        title="Capital IRF Comparison (1% TFP Shock)",
        xlabel="Quarters after shock",
        ylabel="% Deviation from SS / % Shock",
        legend=:topright)
    plot!(plot_irf_comp, t_range, irfs_reiter.K[1:T_plot+1], label="Reiter", lw=2, ls=:dash)
    plot!(plot_irf_comp, t_range, irf_ssj_K[1:T_plot+1], label="SSJ", lw=2, ls=:dot)
    plot!(plot_irf_comp, t_range, irf_ks[1:T_plot+1], label="KS (Simulated)", lw=2.5, ls=:solid, color=:black, alpha=0.7)
    hline!([0], color=:gray, ls=:dash, label="")
    savefig(plot_irf_comp, "IRF_Comparison_1pct.png")
    println("Saved IRF_Comparison_1pct.png")

    # --- 1e. Plot BKM IRF Panel ---
    println("\nPlotting BKM IRF panel...")
    # Top Row
    p_z = plot(t_range, irfs_bkm.Z[1:T_plot+1], title="TFP (Z)", legend=false, lw=2)
    p_k = plot(t_range, irfs_bkm.K[1:T_plot+1], title="Capital (K)", legend=false, lw=2)
    p_kvar = plot(t_range, irfs_bkm.K_var[1:T_plot+1], title="Capital Variance", legend=false, lw=2)
    # Bottom Row
    p_c = plot(t_range, irfs_bkm.C[1:T_plot+1], title="Consumption (C)", legend=false, lw=2)
    p_r = plot(t_range, irfs_bkm.r[1:T_plot+1], title="Interest Rate (r)", legend=false, lw=2)
    p_w = plot(t_range, irfs_bkm.w[1:T_plot+1], title="Wage (w)", legend=false, lw=2)

    # Combine into a panel plot
    plot_bkm_panel = plot(p_z, p_k, p_kvar, p_c, p_r, p_w, layout=(2, 3),
        size=(1200, 600),
        plot_title="BKM Impulse Responses (1% TFP Shock)",
        ylabel="% Dev from SS / % Shock",
        xlabel="Quarters after shock",
        left_margin=5Plots.mm,
        bottom_margin=10Plots.mm) # Added spacing between rows

    savefig(plot_bkm_panel, "BKM_IRF_Panel.png")
    println("Saved BKM_IRF_Panel.png")

    # --- Part 2: Long Simulation and Moment Comparison (1% TFP Shock) ---
    println("\n--- Part 2: Long Simulation (1% TFP Shock) ---")

    # 1. Simulate the BKM economy using the *full* TFP path.
    sim_path_bkm_full = simulate_economy(res_ks.Z_sim_path, irfs_bkm, res_ss, prim_ss, sim_bkm)

    # 2. Simulate the Reiter economy using the same TFP path
    println("\nSimulating Reiter economy...")
    sim_path_reiter_full = simulate_reiter_economy(res_ks.Z_sim_path, system_reiter, 
                                                     g_x_reiter, h_x_reiter, res_ss, prim_ss; ρ=0.75)

    # 3. Extract post-burn-in moments from all simulations.
    burn_in_periods = sim_ks.burn
    
    # KS moments (post-burn-in)
    moments_ks = (
        K = res_ks.K_sim_path[(burn_in_periods+1):end],
        K_var = res_ks.K_var_sim_path[(burn_in_periods+1):end],
        C = res_ks.C_sim_path[(burn_in_periods+1):end],
        r = res_ks.r_sim_path[(burn_in_periods+1):end],
        w = res_ks.w_sim_path[(burn_in_periods+1):end]
    )

    # BKM moments (also post-burn-in)
    moments_bkm = (
        K = sim_path_bkm_full.K[(burn_in_periods+1):end],
        K_var = sim_path_bkm_full.K_var[(burn_in_periods+1):end],
        C = sim_path_bkm_full.C[(burn_in_periods+1):end],
        r = sim_path_bkm_full.r[(burn_in_periods+1):end],
        w = sim_path_bkm_full.w[(burn_in_periods+1):end]
    )

    # Reiter moments (also post-burn-in)
    moments_reiter = (
        K = sim_path_reiter_full.K[(burn_in_periods+1):end],
        K_var = sim_path_reiter_full.K_var[(burn_in_periods+1):end],
        C = sim_path_reiter_full.C[(burn_in_periods+1):end],
        r = sim_path_reiter_full.r[(burn_in_periods+1):end],
        w = sim_path_reiter_full.w[(burn_in_periods+1):end]
    )

    print_moments_table(moments_ks, moments_bkm, moments_reiter, shock)


    # --- Part 3: Analysis with Scaled-Up Shocks ---
    #=
    println("\n--- Part 3: Analysis with Scaled-Up Shocks ---")
    for shock_scaled in shock_sizes[2:end]
        println("\n>>> Processing for a $(shock_scaled*100)% TFP shock...")

        # --- Re-solve for the new shock size ---
        # KS (requires full re-solve as it is a non-linear method)
        println("Re-solving with Krusell-Smith...")
        prim_ks_scaled, sim_ks_scaled, res_ks_scaled = KS.SolveModel(; z_grid=[1.0+shock_scaled, 1.0-shock_scaled], K_min=11.62 - 25*shock_scaled, K_max=11.62 + 25*shock_scaled);
        irf_ks_dev_scaled, _ = KS.estimate_impulse_response(prim_ks_scaled, res_ks_scaled, sim_ks_scaled, num_simulations=10000, max_horizon=T_irf)
        K_bar_ks_scaled = mean(res_ks_scaled.K_sim_path[(sim_ks_scaled.burn+1):end])
        irf_ks_scaled = vcat(0.0, (irf_ks_dev_scaled ./ K_bar_ks_scaled) .* (1 / shock_scaled))[1:T_irf+1]

        # BKM - NOTE: We DO NOT re-solve for a larger shock.
        # The essence of linearization is that the normalized IRF is independent of the shock size.
        # We will use the baseline IRF (`irfs_bkm`) computed from the 1% shock for all comparisons.
        println("For BKM, using baseline IRFs (linearization assumption)...")
        # The IRFs are already stored in the `irfs_bkm` variable from Part 1.
        # The original lines that re-solved the model are removed.


        # 2. Simulate the BKM economy using the full, high-volatility TFP path.
        #    We now use the baseline `irfs_bkm` and `sim_bkm` from the 1% shock solution.
        sim_path_bkm_full_scaled = simulate_economy(res_ks_scaled.Z_sim_path, irfs_bkm, res_ss, prim_ss, sim_bkm)

        # 3. Extract post-burn-in moments from both simulations.
        burn_in_periods_scaled = sim_ks_scaled.burn
        # KS moments from the new full solution (post-burn-in)
        moments_ks_scaled = (
            K = res_ks_scaled.K_sim_path[(burn_in_periods_scaled+1):end],
            K_var = res_ks_scaled.K_var_sim_path[(burn_in_periods_scaled+1):end],
            C = res_ks_scaled.C_sim_path[(burn_in_periods_scaled+1):end],
            r = res_ks_scaled.r_sim_path[(burn_in_periods_scaled+1):end],
            w = res_ks_scaled.w_sim_path[(burn_in_periods_scaled+1):end]
        )

        # BKM moments (now also post-burn-in)
        moments_bkm_scaled = (
            K = sim_path_bkm_full_scaled.K[(burn_in_periods_scaled+1):end],
            K_var = sim_path_bkm_full_scaled.K_var[(burn_in_periods_scaled+1):end],
            C = sim_path_bkm_full_scaled.C[(burn_in_periods_scaled+1):end],
            r = sim_path_bkm_full_scaled.r[(burn_in_periods_scaled+1):end],
            w = sim_path_bkm_full_scaled.w[(burn_in_periods_scaled+1):end]
        )
        print_moments_table(moments_ks_scaled, moments_bkm_scaled, shock_scaled)
    end
    =#
        println("\n--- Part 3: Analysis with Scaled-Up Shocks ---")
    if use_level_deviations
        println("Using LEVEL deviations (to test certainty equivalence)")
        # Recompute BKM IRFs in levels
        irfs_bkm_levels = ComputeIRFs(prim_ss, res_ss, res_tr_bkm, sim_bkm; use_levels=true)
    else
        println("Using PERCENT deviations (standard approach)")
        irfs_bkm_levels = irfs_bkm  # Use the percentage-based IRFs
    end
    #=
    for shock_scaled in shock_sizes[2:end]
        println("\n>>> Processing for a $(shock_scaled*100)% TFP shock...")

        # Re-solve KS
        println("Re-solving with Krusell-Smith...")
        prim_ks_scaled, sim_ks_scaled, res_ks_scaled = KS.SolveModel(
            z_grid=[1.0+shock_scaled, 1.0-shock_scaled], 
            K_min=11.62 - 25*shock_scaled, 
            K_max=11.62 + 25*shock_scaled
        )
    =#
    ρ_tfp = 0.75  # TFP persistence from BKM code


    for shock_scaled in shock_sizes[2:end]
        println("\n>>> Processing for a $(shock_scaled*100)% TFP shock...")

        # Compute the unconditional std dev in log space
        σ_z = shock_scaled #/ sqrt(1 - ρ_tfp^2)
        
        # Create symmetric grid in LOG space (not level space!)
        z_grid_scaled = [exp(-σ_z), exp(σ_z)]
        
        println("  σ_z = $σ_z")
        println("  z_grid = $z_grid_scaled")
        
        # Re-solve KS with proper grid
        println("Re-solving with Krusell-Smith...")
        prim_ks_scaled, sim_ks_scaled, res_ks_scaled = KS.SolveModel(
            z_grid=z_grid_scaled,
            K_min=11.62 - 40*shock_scaled,
            K_max=11.62 + 70*shock_scaled,
            nK=Int(300*shock_scaled),
        )

        # Simulate BKM with appropriate IRFs
        sim_path_bkm_full_scaled = simulate_economy(
            res_ks_scaled.Z_sim_path, 
            irfs_bkm_levels,  # Use level or percent IRFs based on flag
            res_ss, 
            prim_ss, 
            sim_bkm; 
            use_levels=use_level_deviations
        )

        # Simulate Reiter economy
        println("Simulating Reiter economy...")
        sim_path_reiter_full_scaled = simulate_reiter_economy(
            res_ks_scaled.Z_sim_path, 
            system_reiter, 
            g_x_reiter, 
            h_x_reiter, 
            res_ss, 
            prim_ss; 
            ρ=0.75
        )

        # Extract moments
        burn_in_periods_scaled = sim_ks_scaled.burn
        moments_ks_scaled = (
            K = res_ks_scaled.K_sim_path[(burn_in_periods_scaled+1):end],
            K_var = res_ks_scaled.K_var_sim_path[(burn_in_periods_scaled+1):end],
            C = res_ks_scaled.C_sim_path[(burn_in_periods_scaled+1):end],
            r = res_ks_scaled.r_sim_path[(burn_in_periods_scaled+1):end],
            w = res_ks_scaled.w_sim_path[(burn_in_periods_scaled+1):end]
        )

        moments_bkm_scaled = (
            K = sim_path_bkm_full_scaled.K[(burn_in_periods_scaled+1):end],
            K_var = sim_path_bkm_full_scaled.K_var[(burn_in_periods_scaled+1):end],
            C = sim_path_bkm_full_scaled.C[(burn_in_periods_scaled+1):end],
            r = sim_path_bkm_full_scaled.r[(burn_in_periods_scaled+1):end],
            w = sim_path_bkm_full_scaled.w[(burn_in_periods_scaled+1):end]
        )

        moments_reiter_scaled = (
            K = sim_path_reiter_full_scaled.K[(burn_in_periods_scaled+1):end],
            K_var = sim_path_reiter_full_scaled.K_var[(burn_in_periods_scaled+1):end],
            C = sim_path_reiter_full_scaled.C[(burn_in_periods_scaled+1):end],
            r = sim_path_reiter_full_scaled.r[(burn_in_periods_scaled+1):end],
            w = sim_path_reiter_full_scaled.w[(burn_in_periods_scaled+1):end]
        )
        
        print_moments_table(moments_ks_scaled, moments_bkm_scaled, moments_reiter_scaled, shock_scaled)
    end

    println("\nAnalysis complete.")
end

# --- Run the main analysis function ---
main()

