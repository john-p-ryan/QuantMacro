#=
Solves the stochastic growth model using value function iteration
Includes parallelization along the TFP grid
=#


#keyword-enabled structure to hold model primitives
@everywhere @with_kw struct Primitives
    β::Float64 = 0.99                                       #discount rate
    δ::Float64 = 0.025                                      #depreciation rate
    α::Float64 = 0.36                                       #capital share
    k_min::Float64 = 0.01                                   #capital lower bound
    k_max::Float64 = 90.0                                   #capital upper bound
    nk::Int64 = 1000                                        #number of capital grid points
    k_grid::Array{Float64,1} = collect(range(k_min, length = nk, stop = k_max)) #capital grid
    z_grid::Vector{Float64} = [1.25; 0.2]                   #state grid
    nz::Int64 = length(z_grid)                              #number of states
    π::Matrix{Float64} = [0.977 0.023;                      # Transition matrix. Rows are a given state
                          0.074 0.926]
end

#structure that holds model results
@everywhere @with_kw mutable struct Results
    val_func::SharedArray{Float64}                          #value function
    pol_func::SharedArray{Float64}                          #policy function
end

#function for initializing model primitives and results
@everywhere function Initialize()
    prim = Primitives()                                         #initialize primtiives
    val_func = SharedArray{Float64}(prim.nk, prim.nz)           #initial value function guess
    pol_func = SharedArray{Float64}(prim.nk, prim.nz)           #initial policy function guess
    res = Results(val_func, pol_func)                           #initialize results struct
    return prim, res                                            #return deliverables
end

#Bellman Operator
function Bellman(prim::Primitives,res::Results)
    @unpack_Results res
    @unpack_Primitives prim
    
    v_next = SharedArray{Float64}(nk, nz)               # next guess of value function to fill
    kp_next = SharedArray{Float64}(nk, nz)              # next guess of policy function to fill

    @sync @distributed for z_index in 1:nz
        z = z_grid[z_index]

        choice_lower = 1                         #for exploiting monotonicity of policy function
        for (k_index, k) in enumerate(k_grid)
            candidate_max = -Inf #bad candidate max
            budget = z * k^α + (1-δ)*k # budget
            prob_vec = π[z_index,:]    # π is (2x2), rows are current state...cols are future state

            for kp_index in choice_lower:nk #loop over possible selections of k'
                kp = k_grid[kp_index]
                c = budget - kp #consumption given k' selection
                if c>0 #check for positivity
                    val = log(c) + β*(prob_vec ⋅ val_func[kp_index, :]) #compute value
                    if val>candidate_max #check for new max value
                        candidate_max = val #update max value
                        kp_next[k_index, z_index] = kp #update policy function
                        choice_lower = kp_index #update lowest possible choice
                    end
                end
            end
            v_next[k_index, z_index] = candidate_max #update value function
        end
    end

    return v_next, kp_next #return next guess of value function
end

#Value function iteration
function V_iterate(prim::Primitives, res::Results; tol::Float64 = 1e-6)
    err = 100*tol
    n = 0 #counter

    while err>tol #begin iteration
        v_next, kp_next = Bellman(prim, res) #spit out new vectors
        err = maximum(abs.(v_next - res.val_func))#/abs(v_next[prim.nk, 1]) #reset error level
        res.val_func = v_next #update value function
        res.pol_func = kp_next
        n+=1
        if n % 50 == 0
            println("Value function iteration: ", n, " with error: ", err)
        end
    end
    println("Value function converged in ", n, " iterations.")
end

#solve the model
function Solve_model(prim::Primitives, res::Results)
    V_iterate(prim, res) #in this case, all we have to do is the value function iteration!
end
##############################################################################
