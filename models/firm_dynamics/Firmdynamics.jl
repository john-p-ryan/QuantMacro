#=
This code implements a version of the Hopenhayn and Ericson and Pakes models of firm dynamics.

Timing: 
    1. realize shock of productivity
    2. production & pay fixed cost
    3. exit decision and entry for next period

Fixed cost of operation paid in wages (=1) and fixed of entry paid in wages (=1) one period ahead
Uses common parameterization across the 2 models including reduced form demand function

John Ryan
Fall 2024
=#

using LinearAlgebra, Optim, NLsolve, Plots


############################## Model parameters #############################
β::Float64 = 0.96                           # discount factor
c_f::Float64 = 1.9                          # fixed cost of operation
c_e::Float64 = 2.5                          # fixed cost of entry
α::Float64 = 0.5                            # labor share
φ_grid::Vector{Float64} = [.60, 1.0]        # productivity grid
nφ::Int64 = length(φ_grid)                  # number of productivity states
θ::Float64 = .9                             # persistence
F::Array{Float64,2} = [θ (1-θ);(1-θ) θ]     # transition matrix
η::Float64 = .5                             # entrant high probability
ν::Vector{Float64} = [1-η, η]               # initial distribution
d::Float64 = 5.0                            # demand parameter




########################### Competitive example ############################

calc_π_competitive(P::Float64, φ::Float64) = (P * α * φ)^(1 / (1-α)) * (1/α - 1) - c_f

# plot profit functions across price for each φ
P_vec = range(.01, 8, length=100)
π_l = calc_π_competitive.(P_vec, φ_grid[1])
π_h = calc_π_competitive.(P_vec, φ_grid[2])
plot(P_vec, π_l, label="φ low", xlabel="P", ylabel="π(P, φ)", title="Profit Functions", dpi=400)
plot!(P_vec, π_h, label="φ high", ls=:dash)
# horizontal dotted line at 0
hline!([0], ls=:dash, lw=.5, c=:black,label="")
ylims!(-4, 16)
savefig("profit_functions.png")


# plot value function across price for each φ
v_l = π_l
v_h = (π_h + β*(1-θ)*v_l) / (1-β*θ)
plot(P_vec, v_l, label="φ low", xlabel="P", ylabel="V(P, φ)", title="Value Functions", dpi=400)
plot!(P_vec, v_h, label="φ high", ls=:dash)
hline!([0], ls=:dash, lw=.5, c=:black,label="")
ylims!(-6, 16)
savefig("value_functions.png")


# plot the exit incentive conditions - value functions if staying
exit_l = π_l + β * (θ * v_l + (1-θ) * v_h)
exit_h = π_h + β * (θ * v_h + (1-θ) * v_l)
plot(P_vec, exit_l, label="φ low", xlabel="P", ylabel="IC", title="Exit Incentive Conditions", dpi=400)
plot!(P_vec, exit_h, label="φ high", ls=:dash)
# plot vertical line at 3.29 - equilibrium price
# vline!([P], ls=:dash, lw=2, c=:black, label="P*")
hline!([0], ls=:dash, lw=.5, c=:black, label="")
ylims!(-6, 16)
savefig("exit_incentive.png")


# plot entry incentive conditions
entry_vec = η * v_h + (1-η) * v_l .- c_e
plot(P_vec, entry_vec, label="", xlabel="P", ylabel="IC", title="Entry Incentive Conditions", dpi=400)
# plot horizontal line at c_e
# hline!([c_e], ls=:dash, lw=2, c=:black, label="cₑ")
# plot vertical line at 3.29 from -6 to c_e
hline!([0], ls=:dash, lw=.5, c=:black,label="")
scatter!([P], [0], ls=:dash, lw=1, c=:red, label="P*")
ylims!(-8, 18)
savefig("entry_incentive.png")



# solve for equilibrium
function P_residual_competitve(P::Float64)
    π_l = calc_π_competitive(P, φ_grid[1])
    π_h = calc_π_competitive(P, φ_grid[2])
    v_l = π_l
    v_h = (π_h + β*(1-θ)*v_l) / (1-β*θ)
    return (η*v_h + (1-η)*v_l - c_e)^2
end
# plot(P_residual_competitve, 1.0, 6.0, label="", xlabel="P", ylabel="Residual")
opt = optimize(P_residual_competitve, 1.0, 6.0)
P = opt.minimizer
Q = d - P


# equilibrium values
function M_residual(M::Float64)
    μ = [M, η * M / (1-θ)]
    n_policy = (P * α * φ_grid).^(1 / (1 - α))
    Q_s = μ ⋅ (φ_grid .* n_policy.^α)  
    return (Q - Q_s)^2
end
# plot(M_residual, 0.0, 1.0, label="", xlabel="M", ylabel="Residual")
opt = optimize(M_residual, 0.0, 1.0)
M = opt.minimizer
μ = [M, η * M / (1-θ)]
n_policy = (P * α * φ_grid).^(1 / (1 - α))
π_vec = calc_π_competitive.(P, φ_grid) # indirect profit function
v_vec = [π_vec[1], (π_vec[2] + β*(1-θ)*π_vec[1]) / (1-β*θ)]




############################# Monopoly example ############################

n_m_policy = similar(φ_grid)
for (i, φ) in enumerate(φ_grid)
    obj(n) = -((d - φ*n^α) * φ*n^α - n - c_f)
    n_m_policy[i] = optimize(obj, 0.0, (d/φ)^(1/α)).minimizer
end
n_m_policy

π_m = (d .- φ_grid .* n_m_policy.^α) .* φ_grid .* n_m_policy.^α - n_m_policy .- c_f
v_m = zeros(nφ)
for i in 1:10_000
    v_m = π_m + β * F * v_m
end
v_m

Q_m = φ_grid .* n_m_policy.^α
P_m = d .- Q_m




############################## Duopoly example #############################

function duopoly_response(φ_i, φ_j, n_j)
    obj(n) = - ((d - φ_i*n^α - φ_j*n_j^α) * φ_i*n^α - n - c_f)
    n_i = optimize(obj, 0.0, (d/φ_i)^(1/α)).minimizer
    π_i = -obj(n_i)
    return n_i, π_i
end


# find equilibria - fixed points of the response function (asymmetric off path due to exit)

function duopoly_fixed_point(φ_i, φ_j; tol = 1e-6, maxiter = 1000)
    error = 100 * tol
    iter = 0
    n_i, n_j, π_i, π_j = 0.0, 0.0, 0.0, 0.0
    while error > tol && iter < maxiter
        n_i_new, π_i_new = duopoly_response(φ_i, φ_j, n_j)
        n_j_new, π_j_new = duopoly_response(φ_j, φ_i, n_i_new)
        error = maximum([abs(n_i_new - n_i), abs(n_j_new - n_j)])
        n_i, n_j, π_i, π_j = n_i_new, n_j_new, π_i_new, π_j_new
        iter += 1
    end
    if iter == maxiter
        println("Maximum iterations reached in duopoly_fixed_point")
    elseif error < tol
        return n_i, n_j, π_i, π_j
    end
end


n_ll, _, π_ll, _ = duopoly_fixed_point(φ_grid[1], φ_grid[1])
n_lh, n_hl, π_lh, π_hl = duopoly_fixed_point(φ_grid[1], φ_grid[2])
n_hh, _, π_hh, _ = duopoly_fixed_point(φ_grid[2], φ_grid[2])

P_ll = d - 2 * φ_grid[1] * n_ll^α
Q_ll = d - P_ll
P_hh = d - 2 * φ_grid[2] * n_hh^α
Q_hh = d - P_hh
P_lh = d - φ_grid[1] * n_lh^α - φ_grid[2] * n_hl^α
Q_lh = d - P_lh


# plot all 8 response functions n(φ_i, φ_j | n_j)
n_grid = range(.5, 1.5, length=20)
n_1ll = [duopoly_response(φ_grid[1], φ_grid[1], n_j)[1] for n_j in n_grid]
n_1lh = [duopoly_response(φ_grid[1], φ_grid[2], n_j)[1] for n_j in n_grid]
n_1hl = [duopoly_response(φ_grid[2], φ_grid[1], n_j)[1] for n_j in n_grid]
n_1hh = [duopoly_response(φ_grid[2], φ_grid[2], n_j)[1] for n_j in n_grid]
plot(n_grid, n_1ll, label="n₁(φₗ, φₗ | n₂)", xlabel="n₂", ylabel="n₁", title="Response Functions", lw=2, dpi=400)
plot!(n_grid, n_1lh, label="n₁(φₗ, φₕ | n₂)", linestyle=:dash, lw=2)
plot!(n_grid, n_1hl, label="n₁(φₕ, φₗ | n₂)", marker=:circle, markersize=3)
plot!(n_grid, n_1hh, label="n₁(φₕ, φₕ | n₂)", marker=:diamond, markersize=3)
plot!(n_1ll, n_grid, label="n₂(φₗ, φₗ | n₁)", lw = 2)
plot!(n_1lh, n_grid, label="n₂(φₗ, φₕ | n₁)", marker=:circle, markersize=3)
plot!(n_1hl, n_grid, label="n₂(φₕ, φₗ | n₁)", linestyle=:dash, lw=2)
plot!(n_1hh, n_grid, label="n₂(φₕ, φₕ | n₁)", marker=:diamond, markersize=3)
# add a large empty circle at (.949, .949) to indicate the fixed point
scatter!([n_ll], [n_ll], label="fixed point", markersize=8, markercolor=:black, markerstrokecolor=:black)
scatter!([n_hh], [n_hh], label="", markersize=8, markercolor=:black, markerstrokecolor=:black)
scatter!([n_lh], [n_hl], label="", markersize=8, markercolor=:black, markerstrokecolor=:black)
scatter!([n_hl], [n_lh], label="", markersize=8, markercolor=:black, markerstrokecolor=:black)
savefig("response_functions.png")



# solve for the value functions in the duopoly case
#=
v_lh = π_lh # exit
v_ll = π_ll + β * (θ^2 * v_ll + (1-θ)^2 * v_hh + (1-θ) * θ * (v_hl + v_lh))
v_hh = π_hh + β * ((1-θ)^2 * v_ll + θ^2 * v_hh + θ * (1-θ) * (v_hl + v_lh))
v_hl = π_hl + β * (θ * v_h∅ + (1-θ) * v_l∅) # opponent exits
v_h∅ = π_mh + β * (θ * v_h∅ + (1-θ) * v_l∅) # no entry while high shock
v_l∅ = π_ml + β * (θ * (1-η) * v_ll + θ * η * v_lh + (1-θ) * (1-η) * v_hl + (1-θ) * η * v_hh) # entry while low shock
=#

π_ml, π_mh = π_m
function v_system!(F, v)
    v_lh = π_lh # exit
    v_ll, v_hh, v_hl, v_h∅, v_l∅ = v
    F[1] = π_ll + β * (θ^2 * v_ll + (1-θ)^2 * v_hh + (1-θ) * θ * (v_hl + v_lh)) - v_ll
    F[2] = π_hh + β * ((1-θ)^2 * v_ll + θ^2 * v_hh + θ * (1-θ) * (v_hl + v_lh)) - v_hh
    F[3] = π_hl + β * (θ * v_h∅ + (1-θ) * v_l∅) - v_hl # opponent exits
    F[4] = π_mh + β * (θ * v_h∅ + (1-θ) * v_l∅) - v_h∅ # no entry while high shock
    F[5] = π_ml + β * (θ * (1-η) * v_ll + θ * η * v_lh + (1-θ) * (1-η) * v_hl + (1-θ) * η * v_hh) - v_l∅ # entry while low shock
    return F
end

v_lh = π_lh
sol = nlsolve(v_system!, zeros(5))
v_ll, v_hh, v_hl, v_h∅, v_l∅ = sol.zero

# check for positive contuation values
θ^2 * v_ll + (1-θ)^2 * v_hh + (1-θ) * θ * (v_hl + v_lh)
(1-θ)^2 * v_ll + θ^2 * v_hh + θ * (1-θ) * (v_hl + v_lh)
θ * v_h∅ + (1-θ) * v_l∅
θ * (1-η) * v_ll + θ * η * v_lh + (1-θ) * (1-η) * v_hl + (1-θ) * η * v_hh

# check for negative continuation values
θ^2 * v_lh + (1-θ)^2 * v_hl + θ*(1-θ)*(v_ll + v_hh)

# check for entry conditions 
# > c_e
(1-η)*(θ * v_ll + (1-θ)*v_lh) + η*(θ * v_hl + (1-θ)*v_hh)
(1-η)^2 * v_ll + η^2 * v_hh + η * (1-η) * (v_hl + v_lh)
# < c_e
(1-η)*(θ * v_lh + (1-θ)*v_ll) + η*(θ * v_hh + (1-θ)*v_hl)







####################################### non-binding case #######################################

function plot_HMPcost()
    # Define the x-axis range (Number of Firms)
    x = 0:0.1:9

    # Define the entry cost functions (dummy data for demonstration)
    # Non-binding function (red curve)
    #y1 = 8 ./ (x .+ 0.1) .-0.1

    # Binding at Ce, non-binding at C'e function (green curve)
    y2 = 8 ./ (x.-.5) .+ .5

    # Define the cost levels
    ce = 1.5
    ce_prime = 4.5

    # Find the intersection point for the blue lines (approximate)
    nf = 5.0  # This is a visual estimation from the graph. 
    # You might need a numerical solver for a more precise intersection in a real scenario

    # Create the plot
    plot(x, y2, 
        label="vₑ", 
        color=:red, 
        linewidth=2,
        xlims=(0, 9),
        ylims=(0, 9),
        xlabel=" ",
        ylabel="vₑ, κ",
        xticks=false,
        yticks=false,
        title="Entry Cost Function",
        legend=:topright,
        #xguide_position=:right,
        #grid=true,
        aspect_ratio=.7, # Keeps the x and y scales similar
        #framestyle=:box    # Creates a box around the plot
        dpi=400
    )

    #plot!(x, y2, label="Binding at cₑ ,\nNon-binding at c'ₑ", color=:green, linewidth=2)

    # Add horizontal lines for Ce and C'e
    plot!([0, nf], [ce, ce], label="", color=:green, linewidth=2)
    plot!([0, nf], [ce_prime, ce_prime], label="", color=:blue, linestyle=:dash, linewidth=2)

    # Add a vertical line at NF
    # vline!([nf], label="", color=:black, linewidth=2)
    plot!([nf, nf], [ce, 10], label="", color=:black, linewidth=2)

    # Add a vertical dotted line from NF to the red curve
    # plot!([nf, nf], [0, 10/nf], label="", color=:blue, linestyle=:dot, linewidth=2)

    # Add labels for ce, c'e, and NF
    annotate!(0.5, ce + 0.5, text("cₑ", :green, :left, 12))
    annotate!(0.5, ce_prime + 0.5, text("c'ₑ", :blue, :left, 12))
    #annotate!(nf + 0.5, 0.5, text("NF", :blue, :bottom, 12))
    annotate!(9, -0.5, text("# Firms", :right, 12))
    annotate!(nf+.25, -0.5, text("NF", :right, 12))
    # More customizable arrow using arrows=true
    plot!([2.3, 2.5], [3, 4.33], 
      arrow=true,
      color=:black,
      label=false)
    
    annotate!(1.5, 2.7, text("non-binding", :left, 10))

    plot!([6.0, 5.1], [3.2, 2.37], 
    arrow=true,
    color=:black,
    label=false)

    annotate!(5.5, 3.5, text("binding", :left, 10))


    # Display the plot
    display(plot!())
end
plot_HMPcost()
savefig("HMPcost.png")



# solve for value functions in the non-binding case


function v_system_nonbinding!(F, v)
    v_lh = π_lh # exit
    v_ll, v_hh, v_hl, v_h∅, v_l∅ = v
    F[1] = π_ll + β * (θ^2 * v_ll + (1-θ)^2 * v_hh + (1-θ) * θ * (v_hl + v_lh)) - v_ll
    F[2] = π_hh + β * ((1-θ)^2 * v_ll + θ^2 * v_hh + θ * (1-θ) * (v_hl + v_lh)) - v_hh
    F[3] = π_hl + β * (θ * v_h∅ + (1-θ) * v_l∅) - v_hl # opponent exits
    F[4] = π_mh + β * (θ * v_h∅ + (1-θ) * v_l∅) - v_h∅ # no entry while high shock
    F[5] = π_ml + β * (θ * v_l∅ + (1-θ) * v_h∅) - v_l∅ # entry while low shock
    return F
end

v_lh = π_lh
sol = nlsolve(v_system_nonbinding!, zeros(5))
v_ll, v_hh, v_hl, v_h∅, v_l∅ = sol.zero


# check for positive contuation values
v_ll - π_ll
v_hh - π_hh
v_hl - π_hl
v_hm - π_mh
v_lm - π_ml

# check for negative continuation values
θ^2 * v_lh + (1-θ)^2 * v_hl + θ*(1-θ)*(v_ll + v_hh)
π_lh + β * (θ^2 * v_lh + (1-θ)^2 * v_hl + θ*(1-θ)*(v_ll + v_hh))

# check entry condition
(1-η)*(θ*v_ll + (1-θ)*v_lh) + η*(θ*v_hl + (1-θ)*v_hh)