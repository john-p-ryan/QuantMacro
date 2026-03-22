module AiyagariModule
    include("Aiyagari.jl")
end

module KSModule
    include("KS.jl")
end

using Plots, LinearAlgebra,  Parameters, Statistics


# Solve the Aiyagari model
@time prim_Aiyagari, res_Aiyagari = AiyagariModule.SolveModel()

# plot the capital policy function
plot(prim_Aiyagari.k_grid, res_Aiyagari.k_policy, label=["Employed" "Unemployed"], xlabel="k", ylabel="k'", title="Capital Policy Function")

# plot the cross sectional distribution of wealth
plot(prim_Aiyagari.k_grid, sum(res_Aiyagari.μ, dims=2), xlabel="k", ylabel="Density", title="Cross-Sectional Distribution of Wealth")


# mean capital holdings
res_Aiyagari.K

function compute_moments_aiyagari(prim, res)
    @unpack k_grid, ϵ_grid, ē, nϵ, nk, α, δ, L = prim
    @unpack w, μ, K, r = res
    
    
    # Create vectors for all (k, ϵ) combinations
    k_vec = repeat(k_grid, nϵ)
    ϵ_vec = vcat([fill(ϵ_grid[i], nk) for i in 1:nϵ]...)
    μ_vec = vec(μ)
    
    # Compute income for each (k, ϵ) combination
    # income = labor income + capital income
    income_vec = w * ē * ϵ_vec .+ (r - δ) * k_vec
    
    # Compute wealth (capital holdings)
    wealth_vec = k_vec
    
    # Calculate moments
    mean_income = income_vec ⋅ μ_vec
    mean_sq_income = (income_vec .^ 2) ⋅ μ_vec
    var_income = mean_sq_income - mean_income^2
    cv_income = sqrt(var_income) / mean_income
    
    mean_wealth = wealth_vec ⋅ μ_vec
    mean_sq_wealth = (wealth_vec .^ 2) ⋅ μ_vec
    var_wealth = mean_sq_wealth - mean_wealth^2
    cv_wealth = sqrt(var_wealth) / mean_wealth
    
    return cv_income, cv_wealth, var_income, var_wealth, mean_income, mean_wealth
end

cv_income_aiyagari, cv_wealth_aiyagari, var_income_aiyagari, var_wealth_aiyagari, 
    mean_income_aiyagari, mean_wealth_aiyagari = compute_moments_aiyagari(prim_Aiyagari, res_Aiyagari)

println("\n=== AIYAGARI MODEL (Steady State) ===")
println("Mean capital holdings: ", mean_wealth_aiyagari)
println("Variance of wealth: ", var_wealth_aiyagari)
println("CV of wealth: ", cv_wealth_aiyagari)
println("Mean income: ", mean_income_aiyagari)
println("Variance of income: ", var_income_aiyagari)
println("CV of income: ", cv_income_aiyagari)


# Solve the Krusell-Smith model
@time prim_KS, sho_KS, sim_KS, res_KS = KSModule.SolveModel()


using Interpolations

pol_interp = interpolate(res_KS.k_policy, BSpline(Linear()))
pol_interp = scale(pol_interp, 
                   range(prim_KS.k_min, prim_KS.k_max, prim_KS.nk), 
                   1:prim_KS.nϵ, 
                   range(prim_KS.K_min, prim_KS.K_max, prim_KS.nK), 
                   1:prim_KS.nz)

plot(prim_KS.k_grid, pol_interp(prim_KS.k_grid, 1, 11.62, 1), label="Employed, good times", xlabel="k", ylabel="k'", title="Capital Policy Function")
plot!(prim_KS.k_grid, pol_interp(prim_KS.k_grid, 2, 11.62, 1), label="Unemployed, good times")
plot!(prim_KS.k_grid, pol_interp(prim_KS.k_grid, 1, 11.62, 2), label="Employed, bad times")
plot!(prim_KS.k_grid, pol_interp(prim_KS.k_grid, 2, 11.62, 2), label="Unemployed, bad times")


# mean capital holdings over time
println("Expected capital: ", sum(res_KS.K_path[1000:end]) / length(res_KS.K_path[1000:end]))



# income and wealth CV in Krusell-Smith

function compute_panel_moments(prim, res, sim, sho)
    @unpack z_g, z_b, ē, α, δ = prim
    @unpack burn = sim
    
    # Use data after burn-in period
    k_data = res.k_panel[:, burn:end]
    E_data = res.E[:, burn:end]
    K_data = res.K_path[burn:end]
    Z_data = res.Z[burn:end]
    
    T_sim = length(Z_data)
    cv_wealth_vec = zeros(T_sim)
    cv_income_vec = zeros(T_sim)
    var_wealth_vec = zeros(T_sim)
    var_income_vec = zeros(T_sim)
    r_vec = zeros(T_sim)
    w_vec = zeros(T_sim)
    
    for t in 1:T_sim
        # Compute aggregate labor
        if Z_data[t] == z_g
            L = ē * (1 - sho.u_g)
        else
            L = ē * (1 - sho.u_b)
        end
        
        # Compute prices for this period
        r = α * Z_data[t] * (L / K_data[t])^(1 - α) - δ
        w = (1 - α) * Z_data[t] * (K_data[t] / L)^α
        
        r_vec[t] = r
        w_vec[t] = w
        
        # Compute income for each agent at time t
        # income = labor income + capital income
        income_t = w * ē * E_data[:, t] .+ (r - δ) * k_data[:, t]
        
        # Compute wealth for each agent at time t
        wealth_t = k_data[:, t]
        
        # Calculate moments for income
        mean_income_t = mean(income_t)
        var_income_vec[t] = var(income_t)
        cv_income_vec[t] = std(income_t) / mean_income_t
        
        # Calculate moments for wealth
        mean_wealth_t = mean(wealth_t)
        var_wealth_vec[t] = var(wealth_t)
        cv_wealth_vec[t] = std(wealth_t) / mean_wealth_t
    end

    return cv_wealth_vec, cv_income_vec, var_wealth_vec, var_income_vec, r_vec, w_vec
end

cv_wealth_KS, cv_income_KS, var_wealth_KS, var_income_KS, r_KS, w_KS = 
    compute_panel_moments(prim_KS, res_KS, sim_KS, sho_KS)

println("\n=== KRUSELL-SMITH MODEL (With Aggregate Shocks) ===")
println("Mean capital holdings: ", mean(res_KS.K_path[sim_KS.burn:end]))
println("Mean variance of wealth: ", mean(var_wealth_KS))
println("Mean CV of wealth: ", mean(cv_wealth_KS))
println("Mean variance of income: ", mean(var_income_KS))
println("Mean CV of income: ", mean(cv_income_KS))
println("Mean interest rate: ", mean(r_KS))
println("Mean wage: ", mean(w_KS))








# Test the Automatic Stabilizer Hypothesis
using Plots, Statistics, StatsBase

# 1. Correlation Analysis: Do high K periods have lower wealth dispersion?
K_data = res_KS.K_path[sim_KS.burn:end]

println("\nCorrelations:")
println("Corr(K, Var(wealth)): ", cor(K_data, var_wealth_KS))
println("Corr(K, CV(wealth)): ", cor(K_data, cv_wealth_KS))
println("Corr(r, Var(wealth)): ", cor(r_KS, var_wealth_KS))
println("Corr(w, Var(wealth)): ", cor(w_KS, var_wealth_KS))

# 2. Visualize the relationship
p1 = scatter(K_data, var_wealth_KS, 
    xlabel="Aggregate Capital K", 
    ylabel="Variance of Wealth",
    title="K vs Wealth Dispersion",
    legend=false, alpha=0.3, markersize=2)

p2 = scatter(r_KS, var_wealth_KS,
    xlabel="Interest Rate r",
    ylabel="Variance of Wealth", 
    title="Interest Rate vs Wealth Dispersion",
    legend=false, alpha=0.3, markersize=2)

p3 = scatter(w_KS, var_wealth_KS,
    xlabel="Wage w",
    ylabel="Variance of Wealth",
    title="Wage vs Wealth Dispersion", 
    legend=false, alpha=0.3, markersize=2)

plot(p1, p2, p3, layout=(1,3), size=(1200, 400))

# 3. Compare wealth dispersion in good vs bad aggregate states
Z_data = res_KS.Z[sim_KS.burn:end]
good_times = Z_data .== prim_KS.z_g
bad_times = Z_data .== prim_KS.z_b

println("\n=== Wealth Dispersion by Aggregate State ===")
println("Good times (high z):")
println("  Mean K: ", mean(K_data[good_times]))
println("  Mean r: ", mean(r_KS[good_times]))
println("  Mean w: ", mean(w_KS[good_times]))
println("  Mean Var(wealth): ", mean(var_wealth_KS[good_times]))
println("  Mean CV(wealth): ", mean(cv_wealth_KS[good_times]))

println("\nBad times (low z):")
println("  Mean K: ", mean(K_data[bad_times]))
println("  Mean r: ", mean(r_KS[bad_times]))
println("  Mean w: ", mean(w_KS[bad_times]))
println("  Mean Var(wealth): ", mean(var_wealth_KS[bad_times]))
println("  Mean CV(wealth): ", mean(cv_wealth_KS[bad_times]))

# 4. Differential impact on rich vs poor
function wealth_by_quintile(prim, res, sim)
    @unpack burn = sim
    k_data = res.k_panel[:, burn:end]
    T_sim = size(k_data, 2)
    
    quintile_means = zeros(5, T_sim)
    
    for t in 1:T_sim
        wealth_t = k_data[:, t]
        quintiles = quantile(wealth_t, [0.2, 0.4, 0.6, 0.8])
        
        quintile_means[1, t] = mean(wealth_t[wealth_t .<= quintiles[1]])
        quintile_means[2, t] = mean(wealth_t[(wealth_t .> quintiles[1]) .& (wealth_t .<= quintiles[2])])
        quintile_means[3, t] = mean(wealth_t[(wealth_t .> quintiles[2]) .& (wealth_t .<= quintiles[3])])
        quintile_means[4, t] = mean(wealth_t[(wealth_t .> quintiles[3]) .& (wealth_t .<= quintiles[4])])
        quintile_means[5, t] = mean(wealth_t[wealth_t .> quintiles[4]])
    end
    
    return quintile_means
end

quintile_wealth = wealth_by_quintile(prim_KS, res_KS, sim_KS)

# 5. Show how rich vs poor respond to aggregate shocks
println("\n=== Differential Response to K Changes ===")
println("Correlation of quintile wealth with K:")
for q in 1:5
    println("  Quintile $q: ", cor(K_data, quintile_wealth[q, :]))
end

println("\nCorrelation of quintile wealth with r:")
for q in 1:5
    println("  Quintile $q: ", cor(r_KS, quintile_wealth[q, :]))
end

# 6. Plot wealth dispersion dynamics around aggregate shocks
window = 200:600  # Pick an interesting window
p4 = plot(K_data[window], label="Aggregate K", ylabel="K", legend=:topleft)
p5 = plot(var_wealth_KS[window], label="Var(wealth)", ylabel="Variance", 
    legend=:topleft, color=:red)
p6 = plot(r_KS[window], label="r", ylabel="Interest Rate", legend=:topleft, color=:green)

plot(p4, p5, p6, layout=(3,1), size=(800, 600), 
    xlabel="Time", title="Dynamics of K, Wealth Dispersion, and Prices")

# 7. Return to mean effect
println("\n=== Mean Reversion Test ===")
K_high = K_data .> median(K_data)
K_low = K_data .<= median(K_data)

println("When K is HIGH:")
println("  Next period change in Var(wealth): ", 
    mean(diff(var_wealth_KS)[K_high[1:end-1]]))

println("\nWhen K is LOW:")
println("  Next period change in Var(wealth): ", 
    mean(diff(var_wealth_KS)[K_low[1:end-1]]))