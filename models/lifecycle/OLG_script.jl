using Plots, Parameters, LinearAlgebra, Optim, DelimitedFiles

# import functions from OLG_functions.jl
include("OLG_functions.jl")

# read in age - efficiency profile from ef.txt
ef = readdlm("OLG/ef.txt", '\t')[:,1]

plot(ef, xlabel="Model Age", label="", ylabel="Deterministic Efficiency", 
title="Age-Efficiency Profile", lw=1.5, dpi=400)
savefig("OLG/ef.png")


#########################################################################
# With social security
#########################################################################

prim = Primitives()
res = Initialize(prim)
res = V_induction(prim, res)
res = steady_dist(prim, res)
res = market_clearing(prim, res)



#########################################################################
# Without social security
#########################################################################

prim2 = Primitives(θ=0.0, a_max=40.0, na=600)
res2 = Initialize(prim2)
res2 = V_induction(prim2, res2)
res2 = steady_dist(prim2, res2)
res2 = market_clearing(prim2, res2)


# counterfactual: no ss, but prices from with ss
res3= deepcopy(res2)
res3.r = res.r
res3.w = res.w
res3 = V_induction(prim2, res3)
res3 = steady_dist(prim2, res3)


#########################################################################
# plot some policy and value functions
#########################################################################

# value function age 50
plot(prim.a_grid, 
    res.Vʳ[end-16,:], 
    label = "V(a)", 
    title = "Value function for age 50")
xlabel!("Assets")


# value function age 20)
# plot value function for age 20
plot(prim.a_grid, res.Vʷ[20,:,1], label = "z=3.0", title = "Value function for age 20")
plot!(prim.a_grid, res.Vʷ[20,:,2], label = "z=0.5", title = "Value function for age 20")
# update x and y labels
xlabel!("Assets")
ylabel!("V(a)")


# plot policy function for age 20 for both z
plot(prim.a_grid, res.aʷ[20,:,1], label = "z = 3", title = "Policy function for age 20")
plot!(prim.a_grid, res.aʷ[20,:,2], label = "z = 0.5")
# change x label
xlabel!("Assets")
# change y label
ylabel!("a'(a)")




#########################################################################
# distributional analysis
#########################################################################


Fʷ_age_assets = sum(res.Fʷ, dims = 3)[:,:,1]
# join Fʷ_age_assets with Fʳ on age
F_combined = vcat(Fʷ_age_assets, res.Fʳ)
F_collapsed = sum(F_combined, dims = 1)[1,:]
Fʷ_collapsed = sum(Fʷ_age_assets, dims = 1)[1,:]
Fʳ_collapsed = sum(res.Fʳ, dims = 1)[1,:]


Fʷ_age_assets2 = sum(res2.Fʷ, dims = 3)[:,:,1]
F_combined2 = vcat(Fʷ_age_assets2, res2.Fʳ)
F_collapsed2 = sum(F_combined2, dims = 1)[1,:]

Fʷ_age_assets3 = sum(res3.Fʷ, dims = 3)[:,:,1]
F_combined3 = vcat(Fʷ_age_assets3, res3.Fʳ)
F_collapsed3 = sum(F_combined3, dims = 1)[1,:]


function histogram_from_pmf(M, x_grid, k)
    """
    Generates a histogram from a nonlinear probability mass function (PMF).
  
    Args:
      M: An array representing the PMF values.
      x_grid: An array of evenly spaced points corresponding to the PMF.
      k: The desired number of bins in the histogram.
  
    Returns:
      A tuple containing:
        - bin_centers: The centers of the histogram bins.
        - bin_heights: The corresponding heights (probabilities) of the bins.
    """
  
    # 1. Determine bin edges
    bin_width = (x_grid[end] - x_grid[1]) / k
    bin_edges = range(x_grid[1] - bin_width / 2, stop=x_grid[end] + bin_width / 2, length=k + 1)
  
    # 2. Initialize bin heights
    bin_heights = zeros(k)
  
    # 3. Assign PMF values to bins
    for i in eachindex(M)
      # Find the corresponding bin index
      bin_index = searchsortedfirst(bin_edges, x_grid[i]) - 1 
      
      # Handle edge cases (points exactly on bin edges)
      if bin_index < 1
        bin_index = 1
      elseif bin_index > k
        bin_index = k
      end
  
      bin_heights[bin_index] += M[i]
    end
  
    # 4. Normalize bin heights to represent probabilities
    bin_heights ./= sum(M)
  
    # 5. Calculate bin centers
    bin_centers = [0.5 * (bin_edges[i] + bin_edges[i+1]) for i in 1:k]
    
    return bin_centers, bin_heights
end

bins, heights = histogram_from_pmf(F_collapsed, prim.a_grid, 30)
bar(bins, heights, label="", xlabel="Wealth", ylabel="Density", title="Wealth Distribution With SS")
savefig("OLG/wealth_dist.png")

bins2, heights2 = histogram_from_pmf(F_collapsed2, prim2.a_grid, 30)
bar(bins2, heights2, label="", xlabel="Wealth", ylabel="Density", title="Wealth Distribution Without SS", color=:green)
savefig("OLG/wealth_dist2.png")


bins3, heights3 = histogram_from_pmf(F_collapsed3, prim2.a_grid, 30)
bar(bins3, heights3, label="Wealth Distribution", xlabel="Wealth", ylabel="Density", title="Wealth Distribution", legend=:topleft)





#########################################################################
# Inequality measures
#########################################################################


function lorenz_curve(a_grid::Vector{Float64}, F_collapsed::Vector{Float64})
    # Calculate the cumulative sum of wealth and population
    cumulative_wealth = cumsum(a_grid .* F_collapsed)
    cumulative_population = cumsum(F_collapsed)

    # Normalize to get percentages
    total_wealth = cumulative_wealth[end]
    total_population = cumulative_population[end]
    percent_wealth = cumulative_wealth / total_wealth
    percent_population = cumulative_population / total_population

    # Add (0,0) point for plotting
    percent_wealth = [0; percent_wealth]
    percent_population = [0; percent_population]

    return percent_population, percent_wealth
end

function gini_from_lorenz(percent_population, percent_wealth)
  n = length(percent_population)
  area_under_lorenz = 0.0
  for i in 2:n
      area_under_lorenz += 0.5 * (percent_population[i] - percent_population[i-1]) * (percent_wealth[i] + percent_wealth[i-1])
  end
  gini = 1.0 - 2.0 * area_under_lorenz
  return gini
end

function gini_by_age(prim, Fʷ, Fʳ)
  gini_age = zeros(prim.N)
  Fʷ_age_assets = sum(Fʷ, dims = 3)[:,:,1]
  F_combined = vcat(Fʷ_age_assets, Fʳ)
  for i in 1:prim.N
      percent_pop, percent_wealth = lorenz_curve(prim.a_grid, F_combined[i,:])
      gini_age[i] = gini_from_lorenz(percent_pop, percent_wealth)
  end
  return gini_age[2:end]
end

function consumption_gini_by_age(prim::Primitives, res::Results)
  @unpack_Primitives prim
  @unpack_Results res

  # Initialize an array to store Gini coefficients for each age cohort
  gini_coefficients = zeros(N)

  # Function to calculate Gini coefficient for a given distribution
  function gini(distribution, values)
      # Flatten the distribution and values if they are not 1D arrays
      distribution = reshape(distribution, :)
      values = reshape(values, :)

      # Sort values and corresponding distribution by ascending order of values
      sorted_indices = sortperm(values)
      sorted_values = values[sorted_indices]
      sorted_distribution = distribution[sorted_indices]
      
      # Normalize the distribution to ensure it sums up to 1
      sorted_distribution = sorted_distribution / sum(sorted_distribution)

      # Calculate the cumulative distribution
      cumulative_distribution = cumsum(sorted_distribution)

      # Calculate the cumulative distribution of values, weighted by the distribution
      cumulative_values = cumsum(sorted_values .* sorted_distribution)

      # Normalize the cumulative values
      cumulative_values = cumulative_values / cumulative_values[end]

      # Calculate the Gini coefficient using the trapezoidal rule
      gini_coeff = 1.0 - 2.0 * sum((cumulative_values[1:end-1] + cumulative_values[2:end]) .* (cumulative_distribution[2:end] - cumulative_distribution[1:end-1]) / 2.0)
      
      return gini_coeff
  end

  # Gini for workers by age
  for age in 1:Jʳ
      # Extract the wealth distribution for the current age
      wealth_distribution = Fʷ[age, :, :]

      # Extract the consumption distribution for the current age
      consumption_distribution = cʷ[age, :, :]

      # Calculate total wealth for each individual (summing across productivity states)
      # Here we use a simple average of wealth across productivity states
      # Assuming equal probability for each state, which may not be accurate
      # You might want to weight this by the steady state distribution of z if available
      wealth_values = repeat(a_grid', outer = [prim.nz, 1])

      # Calculate consumption for each individual
      consumption_values = repeat(a_grid', outer = [prim.nz, 1])

      # Calculate the Gini coefficient for wealth and consumption
      gini_wealth = gini(wealth_distribution, wealth_values)
      gini_consumption = gini(consumption_distribution, consumption_values)

      gini_coefficients[age] = (gini_wealth + gini_consumption) / 2.0
  end

  # Gini for retirees by age
  for age in 1:(N - Jʳ)
      # Extract the wealth and consumption distributions for the current age
      wealth_distribution = Fʳ[age, :]
      consumption_distribution = cʳ[age, :]

      # Calculate the Gini coefficient for wealth and consumption
      gini_wealth = gini(wealth_distribution, a_grid)
      gini_consumption = gini(consumption_distribution, a_grid)

      gini_coefficients[Jʳ + age] = (gini_wealth + gini_consumption) / 2.0
  end

  return gini_coefficients
end



percent_population, percent_wealth = lorenz_curve(prim.a_grid, F_collapsed)
gini = gini_from_lorenz(percent_population, percent_wealth)
gini_age = gini_by_age(prim, res.Fʷ, res.Fʳ)
consumption_gini = consumption_gini_by_age(prim, res)


percent_pop2, percent_wealth2 = lorenz_curve(prim2.a_grid, F_collapsed2)
gini2 = gini_from_lorenz(percent_pop2, percent_wealth2)
gini_age2 = gini_by_age(prim2, res2.Fʷ, res2.Fʳ)
consumption_gini2 = consumption_gini_by_age(prim2, res2)

percent_pop3, percent_wealth3 = lorenz_curve(prim2.a_grid, F_collapsed3)
gini3 = gini_from_lorenz(percent_pop3, percent_wealth3)
gini_age3 = gini_by_age(prim2, res3.Fʷ, res3.Fʳ)

# Plot the Lorenz curve
plot(percent_population, percent_wealth,
        xlabel="Cumulative Share of Population",
        ylabel="Cumulative Share of Wealth",
        title="Lorenz Curve",
        label="Lorenz Curve with SS",
        legend=:topleft,
        xlims=(0, 1),
        ylims=(0, 1),
        linewidth=2,
        linecolor=:blue,
        dpi=400)

plot!(percent_pop2, percent_wealth2, label="Lorenz Curve without SS", linewidth=2, linecolor=:green)

# Add a 45-degree line for perfect equality
plot!([0, 1], [0, 1], label="Perfect Equality",
    linewidth=2, linestyle=:dash, linecolor=:red)

savefig("OLG/lorenz.png")

plot!(percent_pop3, percent_wealth3, label="Lorenz Curve without SS, but with SS prices", linewidth=2, linecolor=:purple)

plot(2:prim.N, gini_age, label="With Social Security", 
xlabel="Model Age", ylabel="Gini Coefficient", 
title="Gini Coefficient by Age", lw=1.5, dpi=400)

plot!(2:prim2.N, gini_age2, label="Without Social Security", lw=1.5)

savefig("OLG/gini_age.png")

plot!(2:prim2.N, gini_age3, label="Without Social Security, but with SS prices", lw=1.5)

savefig("OLG/gini_age2.png")

plot!(1:prim.N, consumption_gini, label="With Social Security",)
plot!(1:prim2.N, consumption_gini2, label="Without Social Security", lw=1.5)


#########################################################################
# lifecycle profiles
#########################################################################

# plot mean consumption and assets by age
mean_a_age = zeros(prim.N)
mean_a_age2 = zeros(prim2.N)
for i in 1:prim.N
    mean_a_age[i] = sum(prim.a_grid .* F_combined[i,:])
    mean_a_age2[i] = sum(prim2.a_grid .* F_combined2[i,:])
end

plot(1:prim.N, mean_a_age, label="With SS", xlabel="Model Age", ylabel="Mean Assets", title="Mean Assets by Age", lw=1.5, dpi=400)
plot!(1:prim2.N, mean_a_age2, label="Without SS", lw=1.5, legend=:topleft)
savefig("OLG/mean_assets_age.png")

mean_c_age = zeros(prim.N)
mean_c_age2 = zeros(prim2.N)

for i in 1:prim.Jʳ
    for z_index in eachindex(prim.z_grid)
      mean_c_age[i] += sum(res.cʷ[i,:,z_index] .* res.Fʷ[i,:,z_index])
      mean_c_age2[i] += sum(res2.cʷ[i,:,z_index] .* res2.Fʷ[i,:,z_index])
    end
end
for i in 1:prim.N - prim.Jʳ
    mean_c_age[prim.Jʳ + i] = sum(res.cʳ[i,:] .* res.Fʳ[i,:])
    mean_c_age2[prim.Jʳ + i] = sum(res2.cʳ[i,:] .* res2.Fʳ[i,:])
end

plot(1:prim.N, mean_c_age, label="With SS", xlabel="Model Age", ylabel="Mean Consumption", title="Mean Consumption by Age", lw=1.5, dpi=400)
plot!(1:prim2.N, mean_c_age2, label="Without SS", lw=1.5, legend=:topright)
savefig("OLG/mean_consumption_age.png")