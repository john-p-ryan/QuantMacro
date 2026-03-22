using Parameters, LinearAlgebra, Optim, Plots

###### define model primitives ######
β::Float64 = 0.96                           # discount factor
c_f::Float64 = 1.9                          # fixed cost of operation
c_e::Float64 = 3.0                          # fixed cost of entry
α::Float64 = 0.5                            # labor share
φ_grid::Vector{Float64} = [.60, 1.0]        # productivity grid
nφ::Int64 = length(φ_grid)                  # number of productivity states
θ::Float64 = .9                             # persistence
F::Array{Float64,2} = [θ (1-θ);(1-θ) θ]     # transition matrix
η::Float64 = .5                             # entrant high probability
ν::Vector{Float64} = [1-η, η]               # initial distribution
d::Float64 = 5.0                            # demand parameter

x_policy = [1, 0]                           # exit policy = 1 if exit (construction)


calc_π_competitive(P::Float64, φ::Float64) = (P * α * φ)^(1 / (1-α)) * (1/α - 1) - c_f

function free_entry_residual(P::Float64)
    # given price, solve for residual in free entry condition
    π_vec = (P * α * φ_grid).^(1 / (1-α)) * (1/α - 1) .- c_f # indirect profit function
    v = [π_vec[1], (π_vec[2] + β*(1-θ)*π_vec[1])/(1-β*θ)]   # value function
    return (η * v[2] + (1-η) * v[1] - c_e) ^ 2
end


P = optimize(free_entry_residual, 0.0, d).minimizer
Q = d - P
n_policy = (P * α * φ_grid).^(1 / (1 - α))
π_vec = (P * α * φ_grid).^(1 / (1-α)) * (1/α - 1) .- c_f # indirect profit function
v = [π_vec[1], (π_vec[2] + β*(1-θ)*π_vec[1])/(1-β*θ)] # value function
M = Q / (φ_grid ⋅ (n_policy .^α .* [1, η/(1-θ)])) # measure of entrants
μ = [M, η*M/(1-θ)]





# make plots of profit function, value function, exit incentive and entry incentive

P_vec = range(.01, 8, length=100)
π_l = calc_π_competitive.(P_vec, φ_grid[1])
π_h = calc_π_competitive.(P_vec, φ_grid[2])
plot(P_vec, π_l, label="φ low", xlabel="P", ylabel="π(P, φ)", title="Profit Functions", dpi=400)
plot!(P_vec, π_h, label="φ high", ls=:dash)
hline!([0], ls=:dash, c=:black,label="", lw=.5)
ylims!(-4, 16)
savefig("profit_functions.png")


v_l = π_l
v_h = (π_h + β*(1-θ)*π_l)/(1-β*θ)
plot(P_vec, v_l, label="φ low", xlabel="P", ylabel="V(P, φ)", title="Value Functions", dpi=400)
plot!(P_vec, v_h, label="φ high", ls=:dash)
hline!([0], ls=:dash, c=:black,label="", lw=.5)
ylims!(-6, 16)
savefig("value_functions_Hopenhayn.png")


exit_l = θ * v_l + (1-θ) * v_h
exit_h = (1-θ) * v_l + θ * v_h
plot(P_vec, exit_l, label="φ low", xlabel="P", ylabel="IC", title="Exit Incentive", dpi=400)
plot!(P_vec, exit_h, label="φ high", ls=:dash)
hline!([0], ls=:dash, c=:black,label="", lw=.5)
ylims!(-6, 16)
savefig("exit_incentive_Hopenhayn.png")


entry = η * v_h + (1-η) * v_l .- c_e
plot(P_vec, entry, label="Entry Incentive", xlabel="P", ylabel="IC", title="Entry Incentive", dpi=400)
ylims!(-6, 16)
hline!([0], ls=:dash, c=:black,label="", lw=.5)
savefig("entry_incentive_Hopenhayn.png")