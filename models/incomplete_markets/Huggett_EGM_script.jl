# Applications of the Huggett model solved with EGM: the baseline economy, the welfare gain from complete markets,
# and counterfactuals with a tighter borrowing constraint and longer unemployment spells. Figures are saved next to
# this script.
using LinearAlgebra, Plots
using Spline # custom spline package

module EGM
    using Parameters, LinearAlgebra, Optim, SparseArrays, Spline
    include("Huggett_EGM.jl")
end


# Bars for the distribution of wealth a + s among the employed (s_index = 1) or unemployed (s_index = 2)
function wealth_bars!(plt, prim, res, s_index; kwargs...)
    color = s_index == 1 ? :blue : :red
    return bar!(plt, prim.a_hist .+ prim.s_grid[s_index], res.μ[:, s_index]; bar_width=minimum(diff(prim.a_hist)),
                alpha=0.95, color=color, linecolor=color, linewidth=0.1, kwargs...)
end

# Wealth distribution of the employed and unemployed in one figure
function wealth_plot(prim, res; kwargs...)
    plt = plot(; xlabel="Wealth", ylabel="Fraction of the Population", dpi=300, size=(800, 450),
               left_margin=8Plots.mm, kwargs...)
    wealth_bars!(plt, prim, res, 1; label="Employed")
    return wealth_bars!(plt, prim, res, 2; label="Unemployed")
end

# 2x2 figure of the wealth distributions of the employed and unemployed, in the baseline and in a counterfactual
function wealth_comparison(prim, res, prim_cf, res_cf, cf_name)
    panels = []
    for (name, p, r) in (("Benchmark", prim, res), (cf_name, prim_cf, res_cf))
        for (s_index, status) in enumerate(("Employed", "Unemployed"))
            panel = plot(xlabel="Wealth", ylabel="Fraction of Population", xlims=(-1.6, 2.5),
                         title="$name, $status Wealth Distribution")
            push!(panels, wealth_bars!(panel, p, r, s_index; label=""))
        end
    end
    return plot(panels..., layout=(2, 2), size=(1600, 900), dpi=300, left_margin=8Plots.mm, bottom_margin=8Plots.mm)
end


###########################################################
# Baseline
# Solve the Huggett model
@time prim, res = EGM.solve_model(a_min=-2.0, a_max=2.0, na=200, n_hist=2000)

res.q # bond price
# verify market clearing and distribution
res.A # aggregate assets (should be zero)
sum(res.μ) # should be 1

# check the annual interest rate, for a quarterly model period
r = (1 / res.q)^4 - 1


# plot wealth distribution
wealth_plot(prim, res; title="Baseline Wealth Distribution", legend=(0.3, 0.9), xlims=(-1.6, 2.5))
savefig(joinpath(@__DIR__, "huggett_wealth_distribution.png"))


# what fraction are hand to mouth?
sum(res.μ[1, :])

# what fraction of unemployed are hand to mouth?
res.μ[1, 2] / prim.unemp

# plot the marginal wealth distribution
μ_marg = vec(sum(res.μ, dims=2))
plot(prim.a_hist, μ_marg, label="", xlabel="Assets", ylabel="Probability Mass", title="Marginal Wealth Distribution")


# plot the asset policy function
plot(prim.a_grid, res.a_policy, label=["Employed" "Unemployed"], xlabel="a", ylabel="a'", title="Asset Policy Function", lw=2)
plot!(prim.a_grid, prim.a_grid, linestyle=:dash, label="45-degree line", lw=2, color=:black, dpi=300)
savefig(joinpath(@__DIR__, "huggett_asset_policy.png"))

# plot the savings function a' - a
plot(prim.a_grid, res.a_policy .- prim.a_grid, label=["Employed" "Unemployed"], xlabel="a", ylabel="a'-a", title="Savings Function")
# plot 0 line
hline!([0.0], linestyle=:dash, label="Zero Savings", color=:black)

# find the first value where a' - a < 0 for employed agents
savings_employed = res.a_pol_hist[:, 1] .- prim.a_hist
first_negative_index = findfirst(savings_employed .< 0.0)
a_upper = prim.a_hist[first_negative_index]


# plot the value function
plot(prim.a_grid, res.V, label=["Employed" "Unemployed"], xlabel="Assets a", ylabel="V(a,s)", title="Value Function", lw=2, dpi=300)
savefig(joinpath(@__DIR__, "huggett_value_function.png"))

# plot the consumption policy function
plot(prim.a_grid, res.c_policy, label=["Employed" "Unemployed"], xlabel="a", ylabel="c", title="Consumption Policy Function")


res.gini # Gini coefficient

# plot lorenz curve
plot(res.cum_pop, res.lorenz, lw=2, xlabel="Fraction of Agents", ylabel="Cumulative Wealth Share", title="Lorenz Curve", label="")
plot!([0,1], [0,1], linestyle=:dash, label="Line of Equality", color=:black, lw=2, dpi=300)
savefig(joinpath(@__DIR__, "huggett_lorenz_curve.png"))


###################################################################
# Counterfactuals
# Consumption equivalent λ(a, s): the permanent proportional change in consumption that makes a household with value
# V(a, s) as well off as expected welfare W_new. Scaling consumption by 1 + λ under the normalized CRRA utility
# (c^(1-γ) - 1) / (1-γ) maps V + B to (1 + λ)^(1-γ) (V + B), with B = 1 / ((1-β)(1-γ)).
function calculate_λ(W_new, V, prim)
    (; β, γ) = prim
    if γ == 1
        return expm1.((1 - β) .* (W_new .- V))
    end
    B = 1 / ((1 - β) * (1 - γ))
    return ((W_new + B) ./ (V .+ B)) .^ (1 / (1 - γ)) .- 1
end


##################################################################
# calculate complete markets welfare: with full insurance, every household consumes average earnings
μ_e = 1 - prim.unemp
μ_u = prim.unemp
W_cm = EGM.u(μ_e * prim.s_grid[1] + μ_u * prim.s_grid[2], prim.γ) / (1 - prim.β)
res.W

λ_cm = calculate_λ(W_cm, res.V_hist, prim)
WG = res.μ ⋅ λ_cm # welfare gain from complete markets

# plot consumption equivalent gains
plot(prim.a_hist, λ_cm, label=["Employed" "Unemployed"], xlabel="Assets", ylabel="Consumption Equivalent", lw=2, dpi=300)
savefig(joinpath(@__DIR__, "huggett_consumption_equivalents.png"))

# find the fraction of agents with positive consumption equivalent gain (ex post)
sum(res.μ[λ_cm .> 0.0])



################################################################
# tighten borrowing constraint to a_min = -1.0
prim_tight, res_tight = EGM.solve_model(a_min=-1.0, a_max=2.0, na=250, n_hist=2500)
res_tight.q # bond price
# check interest rate
r_tight = (1 / res_tight.q)^4 - 1

# plot the wealth distribution
wealth_plot(prim_tight, res_tight; title="Wealth Distribution (a̲ = -1.0)", legend=:topleft, xlims=(-0.6, 1.5))

# plot the asset policy function
plot(prim_tight.a_grid, res_tight.a_policy, label=["Employed" "Unemployed"], xlabel="a", ylabel="a'")
plot!(prim_tight.a_grid, prim_tight.a_grid, linestyle=:dash, label="45-degree line", color=:black)

# fraction hand to mouth
sum(res_tight.μ[1, :])
# fraction of unemployed hand to mouth
res_tight.μ[1, 2] / prim_tight.unemp

# plot the savings function a' - a
plot(prim_tight.a_grid, res_tight.a_policy .- prim_tight.a_grid, label=["Employed" "Unemployed"], xlabel="a", ylabel="a' - a", title="Savings Policy (a̲ = -1.0)")
# plot 0 line
hline!([0.0], linestyle=:dash, label="Zero Savings", color=:black)
# find the first value where a' - a < 0 for employed agents
savings_employed_tight = res_tight.a_pol_hist[:, 1] .- prim_tight.a_hist
first_negative_index_tight = findfirst(savings_employed_tight .< 0.0)
a_upper_tight = prim_tight.a_hist[first_negative_index_tight]

# calculate consumption equivalents for moving to the baseline a_min = -2.0
λ_tight = calculate_λ(res.W, res_tight.V_hist, prim)
WG_tight = res_tight.μ ⋅ λ_tight # welfare gain from loosening borrowing constraint
# calculate WG for employed and unemployed separately
WG_tight_employed = sum(res_tight.μ[:, 1] .* λ_tight[:, 1]) / μ_e
WG_tight_unemployed = sum(res_tight.μ[:, 2] .* λ_tight[:, 2]) / μ_u


# gini coefficient
res_tight.gini


# 2x2 figure for counterfactual wealth distribution comparison
wealth_comparison(prim, res, prim_tight, res_tight, "Tight Constraint")
savefig(joinpath(@__DIR__, "huggett_wealth_comparison_borrowing.png"))



################################################################
# longer unemployment duration: the unemployed stay unemployed with probability 0.75 instead of 0.5, so spells last
# 4 periods on average. The job-loss probability is unchanged, so the unemployment rate rises.
Π_long = copy(prim.Π) # copy, so the baseline primitives are not modified
Π_long[2, :] = [0.25, 0.75]

prim_long, res_long = EGM.solve_model(a_max=2.0, na=250, n_hist=2500, Π=Π_long)
res_long.q # bond price

# check interest rate
r_long = (1 / res_long.q)^4 - 1

# plot the wealth distribution
wealth_plot(prim_long, res_long; title="Wealth Distribution (Longer Unemployment Duration)", legend=:topright, xlims=(-1.6, 2.5))

# fraction unemployed
sum(res_long.μ[:, 2])

# fraction hand to mouth
sum(res_long.μ[1, :])
# fraction of unemployed hand to mouth
res_long.μ[1, 2] / sum(res_long.μ[:, 2])

# plot the asset policy function
plot(prim_long.a_grid, res_long.a_policy, label=["Employed" "Unemployed"], xlabel="a", ylabel="a'")
plot!(prim_long.a_grid, prim_long.a_grid, linestyle=:dash, label="45-degree line", color=:black)

# plot the savings function a' - a
plot(prim_long.a_grid, res_long.a_policy .- prim_long.a_grid, label=["Employed" "Unemployed"], xlabel="a", ylabel="a' - a")
# plot 0 line
hline!([0.0], linestyle=:dash, label="Zero Savings", color=:black)
# find the first value where a' - a < 0 for employed agents
savings_employed_long = res_long.a_pol_hist[:, 1] .- prim_long.a_hist
first_negative_index_long = findfirst(savings_employed_long .< 0.0)
a_upper_long = prim_long.a_hist[first_negative_index_long]

# calculate consumption equivalents for moving to the baseline, with spells of 2 periods on average
λ_long = calculate_λ(res.W, res_long.V_hist, prim)
WG_long = res_long.μ ⋅ λ_long # welfare gain from shorter unemployment duration
# calculate WG for employed and unemployed separately
WG_long_employed = sum(res_long.μ[:, 1] .* λ_long[:, 1]) / (1 - sum(res_long.μ[:, 2]))
WG_long_unemployed = sum(res_long.μ[:, 2] .* λ_long[:, 2]) / sum(res_long.μ[:, 2])

# gini coefficient
res_long.gini

# make 2x2 figure for unemployment duration wealth distribution comparison
wealth_comparison(prim, res, prim_long, res_long, "Longer Unemployment Duration")
savefig(joinpath(@__DIR__, "huggett_wealth_comparison_unemployment.png"))


##########################################################
# vary a_min more finely and plot q vs a_min
a_min_values = -2.5:0.04:-0.9
sweep = [EGM.solve_model(a_min=a_min, a_max=3.0, na=200, n_hist=2000) for a_min in a_min_values]
q_values = [res_i.q for (_, res_i) in sweep]

# plot q vs a_min
plot(a_min_values, q_values, xlabel="Borrowing Constraint a̲", ylabel="Bond Price q", title="Bond Price vs Borrowing Constraint", label="q(a̲)")

# plot gini vs a_min
gini_values = [res_i.gini for (_, res_i) in sweep]
plot(a_min_values, gini_values, xlabel="Borrowing Constraint a̲", ylabel="Gini Coefficient", title="Gini Coefficient vs Borrowing Constraint", label="Gini(a̲)")


# make a GIF plotting the wealth distribution as a_min varies
@gif for (a_min, (prim_i, res_i)) in zip(a_min_values, sweep)
    plot(prim_i.a_hist, res_i.μ, xlabel="Assets", label=["Employed" "Unemployed"], ylabel="Probability Mass", title="Wealth Distribution (a̲ = $(round(a_min, digits=2)))", ylim=(0, maximum(res_i.μ)*1.1))
    xlims!(-3, 2)
end fps=2
