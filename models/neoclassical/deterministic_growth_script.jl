using Parameters, Plots #import the libraries we want
include("deterministic_growth_functions.jl") #import the functions that solve our growth model
prim, res = Initialize() #initialize primitive and results structs
@time Solve_model(prim, res) #solve the model!
@unpack val_func, pol_func = res
@unpack k_grid = prim

##############Make plots
#value function
plot(k_grid, val_func, title="Value Function V(K)",ylabel = "value V(K)", label = "",xlabel = "capital K")
savefig("Value_Functions.png")

#policy functions
plot(k_grid, pol_func, title="Policy Function K'(K)",ylabel = "policy K'(K)", label = "policy K'(K)",xlabel = "capital K",color="blue",linestyle=:solid)
plot!(k_grid,k_grid,label = "45 degree",color="red",linestyle=:dash)
savefig("Policy_Functions.png")

#changes in policy function
pol_func_δ = pol_func.-k_grid
plot(k_grid, pol_func_δ, title="Saving Policy Function K'(K) - K",ylabel = "saving policy K'(K) - K", label = "",xlabel = "capital K")
hline!([0], linestyle=:dash, color=:black, label="")
savefig("Policy_Functions_Changes.png")

println("All done!")
################################
