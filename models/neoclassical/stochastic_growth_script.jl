using Distributed
addprocs(2)
@everywhere using Parameters, LinearAlgebra, SharedArrays, Distributed
include("stochastic_growth_functions.jl")


@everywhere prim, res = Initialize() #initialize primitive and results structs
@time Solve_model(prim, res) #solve the model!
@unpack val_func, pol_func = res
@unpack k_grid = prim

############## Make plots
#value function
using Plots
plot(k_grid, val_func[:,1], label="Z = 1.25",ylabel = "value V(K)",xlabel = "capital K")
plot!(k_grid, val_func[:,2], label="Z = 0.2",ylabel = "value V(K)",xlabel = "capital K")


plot(k_grid, pol_func[:,1], label="Z = 1.25",ylabel = "policy K'(K)", xlabel = "capital K",color="blue",linestyle=:solid)
plot!(k_grid, pol_func[:,2], label="Z = 0.2",ylabel = "policy K'(K)", xlabel = "capital K",color="red",linestyle=:solid)
plot!(k_grid,k_grid,label = "45 degree",color="black",linestyle=:dash)

plot(k_grid, pol_func[:,1] .- k_grid, label="Z = 1.25",ylabel = "saving policy K'(K) - K", xlabel = "capital K",color="blue",linestyle=:solid)
plot!(k_grid, pol_func[:,2] .- k_grid, label="Z = 0.2",ylabel = "saving policy K'(K) - K", xlabel = "capital K",color="red",linestyle=:solid)
