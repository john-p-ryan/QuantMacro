using LinearAlgebra, Optim, Parameters, Plots, Random, Statistics
Random.seed!(100)


# use this to generate data from "structural model" of AR(1) process
function generate_ar1(ρ::Float64, σ::Float64, ε_shocks::Vector{Float64})
    T = length(ε_shocks)
    ε = ε_shocks * σ
    y = zeros(T)
    y[1] = ε[1]
    for t in 2:T
        y[t] = ρ * y[t-1] + ε[t]
    end
    return y
end


# generate "true" data - keep this fixed
T = 200
ρ_true = 0.5
σ_true = 1.0
shocks = randn(T);
y_true = generate_ar1(ρ_true, σ_true, shocks); # generate "true" data

AR1_reg(y::Vector{Float64}) = (y[2:end]⋅y[1:end-1]) / (y[1:end-1]⋅y[1:end-1]) # OLS estimate of ρ w/ no constant term
ρ̂ = AR1_reg(y_true) # OLS estimate of ρ w/ no constant term


function estimate_ma_N(y, N)
    # estimates the lag coefficients for MA(N) model by least squares

    
    function sse(θ)     # Function to compute the sum of squared errors
        # θ is a vector of length N containing the MA coefficients
        T = length(y)
        y_hat = zeros(T)
        ε = zeros(T)

        for t in 1:T
            ε[t] = y[t]
            for j in 1:min(t-1, N)
                ε[t] -= θ[j] * ε[t-j]
            end
        end

        for t in 1:(T-1)
            for j in 1:min(t, N)
                y_hat[t+1] += θ[j] * ε[t-j+1]
            end
        end

        errors = y - y_hat
        return errors ⋅ errors
        
    end

    # Initial parameter estimates
    initial_params = zeros(N)

    # Optimization
    result = optimize(sse, initial_params, LBFGS())
    
    # Extract the estimated parameters
    θ_hat = result.minimizer
    
    return θ_hat
end

MA1_hat = estimate_ma_N(y_true, 1) # estimate MA(1) model
MA2_hat = estimate_ma_N(y_true, 2) # estimate MA(2) model
MA3_hat = estimate_ma_N(y_true, 3) # estimate MA(3) model

ŝ = std(y_true) # estimate of the variance of the data
# note that s² = σ² / (1 + ρ²)



function indirect_inference(m̂; N, H=1000, T=200)
    # m̂ is a vector of moments from auxiliary model estimated on data
    # N is the number of lags in the MA auxiliary model
    # H is the number of simulations

    simulated_shocks = randn(H, T)

    function obj(x)
        # θ is a vector of length N containing the MA coefficients
        ρ, σ = x[1], x[2]
        m̃ = zeros(H, N+1)
        for sim in 1:H
            sim_data = generate_ar1(ρ, σ, simulated_shocks[sim, :])
            m̃[sim, 1:end-1] = estimate_ma_N(sim_data, N)
            m̃[sim, end] = std(sim_data)
        end
        m̃_mean = mean.(eachcol(m̃))
        return (m̃_mean - m̂) ⋅ (m̃_mean - m̂)
    end

    result = optimize(obj, [0.0, 1.0], LBFGS())
    return result.minimizer
end


m̂1 = vcat(MA1_hat, ŝ)
indirect_inference(m̂1; N=1, H=1000, T=200)

m̂2 = vcat(MA2_hat, ŝ)
indirect_inference(m̂2; N=2, H=1000, T=200)

m̂3 = vcat(MA3_hat, ŝ)
indirect_inference(m̂3; N=3, H=1000, T=200)