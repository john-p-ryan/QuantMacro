using Random, Statistics, Optim, Plots, LinearAlgebra

function generate_ma1_data(θ; ε_shocks, σ=1)
    # θ: MA(1) coefficient
    # ε_shocks: N(0,1) shocks
    # σ: standard deviation of the white noise
    # returns y: MA(1) data

    T = length(ε_shocks)
    ε = ε_shocks * σ

    # Initialize the data array
    y = zeros(T)
    y[1] = ε[1]

    # Generate MA(1) data
    for t in 2:T
        y[t] = ε[t] + θ * ε[t-1]
    end

    return y
end


#=
function AR1_reg(x; suppress_constant=false, SE=false)
    if suppress_constant
        X_lag = x[1:end-1]
    else
        X_lag = hcat(ones(length(x)-1), x[1:end-1])
    end
    X = x[2:end]
    ρ̂ = (X_lag'X_lag) \ (X_lag'X)
    if SE
        ϵ̂ = X - X_lag * ρ̂
        σ̂² = ϵ̂⋅ϵ̂ / (length(x) - length(ρ̂))
        S = σ̂² * inv(X_lag'X_lag)
        se = sqrt.(diag(S))
        return ρ̂, se
    else    
        return ρ̂
    end
end
=#


function AR1_reg(x; suppress_constant::Bool=false, SE::Bool=false)
    T = length(x)
    n = T - 1

    if suppress_constant
        # Regress x_t on x_{t-1}
        # Accumulate sufficient statistics in one pass
        sxx = zero(eltype(x))
        sxy = zero(eltype(x))
        y2  = zero(eltype(x))  # only used if SE==true
        @inbounds @simd for t in 2:T
            xtm1 = x[t-1]; yt = x[t]
            sxx += xtm1*xtm1
            sxy += xtm1*yt
            if SE; y2 += yt*yt; end
        end
        β1 = sxy / sxx
        if !SE
            return [β1]
        else
            dof = n - 1
            @assert dof > 0 "Not enough df for standard errors (need n > 1)"
            rss = y2 - β1*sxy
            σ2  = rss / dof
            se1 = sqrt(σ2 / sxx)
            return [β1], [se1]
        end
    else
        # Regress x_t on (1, x_{t-1})
        # Sufficient stats
        sx = zero(eltype(x))
        sy = zero(eltype(x))
        sxx = zero(eltype(x))
        sxy = zero(eltype(x))
        y2  = zero(eltype(x))  # only used if SE==true
        @inbounds @simd for t in 2:T
            xtm1 = x[t-1]; yt = x[t]
            sx  += xtm1
            sy  += yt
            sxx += xtm1*xtm1
            sxy += xtm1*yt
            if SE; y2 += yt*yt; end
        end
        a = n
        b = sx
        c = sxx
        Δ = a*c - b*b
        @assert Δ > 0 "Design not full rank for AR(1) with intercept"
        # Closed-form 2x2 solve: β = G^{-1} g
        β0 = (c*sy - b*sxy) / Δ
        β1 = (-b*sy + a*sxy) / Δ
        if !SE
            return [β0, β1]
        else
            dof = n - 2
            @assert dof > 0 "Not enough df for standard errors (need n > 2)"
            rss = y2 - (β0*sy + β1*sxy)
            σ2  = rss / dof
            # diag(inv(G)) for [[a b]; [b c]] is [c, a] / Δ
            se0 = sqrt(σ2 * (c / Δ))
            se1 = sqrt(σ2 * (a / Δ))
            return [β0, β1], [se0, se1]
        end
    end
end


function indirect_inference(ρ_hat; H=1000, T=200)
    simulated_shocks = randn(H, T)
    k = length(ρ_hat)
    
    function obj(b)
        ρ̃ = zeros(H, k)
        for sim in 1:H
            sim_data = generate_ma1_data(b; ε_shocks = simulated_shocks[sim, :])
            ρ̃[sim, :] = AR1_reg(sim_data) # estimate auxiliary model
        end
        ρ_mean = mean.(eachcol(ρ̃))
        return (ρ_mean - ρ_hat) ⋅ (ρ_mean - ρ_hat)
    end

    result = optimize(obj, -0.99, 0.99)
    return result.minimizer
end



########### set true data and main estimates ###########
# set seed 
Random.seed!(1000)
shocks = randn(200)
b_0 = -0.5
true_data = generate_ma1_data(b_0; ε_shocks=shocks)
b̂ = cov(true_data[2:end], true_data[1:end-1])
ρ̂, se = AR1_reg(true_data; SE=true)

indirect_inference(ρ̂)





############ simulations with AR(1) estimates ####################

H=1000
T=200
simulated_shocks = randn(H, T)
ρ̃_mat = zeros(H, 2)
se_mat = zeros(H, 2)

for sim in 1:H
    sim_data = generate_ma1_data(b̂ ; ε_shocks = simulated_shocks[sim, :])
    ρ̃, se = AR1_reg(sim_data; SE=true)
    ρ̃_mat[sim, :] = ρ̃
    se_mat[sim, :] = se
end
ρ_mean = mean.(eachcol(ρ̃_mat)) # mean at data estimate b̂
se_mean = mean.(eachcol(se_mat)) # mean of standard errors


for sim in 1:H
    sim_data = generate_ma1_data(b_0 ; ε_shocks = simulated_shocks[sim, :])
    ρ̃, se = AR1_reg(sim_data; SE=true)
    ρ̃_mat[sim, :] = ρ̃
    se_mat[sim, :] = se
end
ρ_mean = mean.(eachcol(ρ̃_mat)) # mean at true parameter b_0
se_mean = mean.(eachcol(se_mat)) # mean of standard errors

std.(eachcol(ρ̃_mat)) # standard deviation of ρ̃ estimates








############## AR(2) model ####################

#= simple, slow version
function AR2_reg(x; suppress_constant=false, SE=false)
    if suppress_constant
        X_lag = hcat(x[1:end-2], x[2:end-1])
    else
        X_lag = hcat(ones(length(x)-2), x[1:end-2], x[2:end-1])
    end
    X = x[3:end]
    ρ̂ = (X_lag'X_lag) \ (X_lag'X)
    if SE
        ϵ̂ = X - X_lag * ρ̂
        σ̂² = ϵ̂⋅ϵ̂ / (length(x) - length(ρ̂))
        S = σ̂² * inv(X_lag'X_lag)
        se = sqrt.(diag(S))
        return ρ̂, se
    else
        return ρ̂
    end
end
=#


function AR2_reg(x; suppress_constant::Bool=false, SE::Bool=false)
    T = length(x)
    n = T - 2

    # Define z1 = x_{t-2}, z2 = x_{t-1}, y = x_t (t = 3..T)
    if suppress_constant
        # Regress y on (z1, z2)  — column order matches your original: (x_{t-2}, x_{t-1})
        s11 = zero(eltype(x))  # z1'z1
        s22 = zero(eltype(x))  # z2'z2
        s12 = zero(eltype(x))  # z1'z2
        s1y = zero(eltype(x))  # z1'y
        s2y = zero(eltype(x))  # z2'y
        y2  = zero(eltype(x))  # y'y (only if SE)

        @inbounds @simd for t in 3:T
            z1 = x[t-2]; z2 = x[t-1]; y = x[t]
            s11 += z1*z1
            s22 += z2*z2
            s12 += z1*z2
            s1y += z1*y
            s2y += z2*y
            if SE; y2 += y*y; end
        end

        Δ = s11*s22 - s12*s12
        @assert Δ > 0 "Design not full rank for AR(2) (no intercept)"
        β1 = ( s22*s1y - s12*s2y) / Δ   # coeff on x_{t-2}
        β2 = ( s11*s2y - s12*s1y) / Δ   # coeff on x_{t-1}

        if !SE
            return [β1, β2]
        else
            dof = n - 2
            @assert dof > 0 "Not enough df for standard errors (need n > 2)"
            # RSS = y'y - β'g  with g = [s1y, s2y]
            rss = y2 - (β1*s1y + β2*s2y)
            σ2  = rss / dof
            # diag(inv(G)) for [[s11 s12];[s12 s22]] is [s22, s11] / Δ
            se1 = sqrt(σ2 * (s22 / Δ))
            se2 = sqrt(σ2 * (s11 / Δ))
            return [β1, β2], [se1, se2]
        end

    else
        # Regress y on (1, z1, z2)
        n1 = n  # number of rows
        s1  = zero(eltype(x))  # sum z1
        s2  = zero(eltype(x))  # sum z2
        sy  = zero(eltype(x))  # sum y
        s11 = zero(eltype(x))  # z1'z1
        s22 = zero(eltype(x))  # z2'z2
        s12 = zero(eltype(x))  # z1'z2
        s1y = zero(eltype(x))  # z1'y
        s2y = zero(eltype(x))  # z2'y
        y2  = zero(eltype(x))  # y'y (only if SE)

        @inbounds @simd for t in 3:T
            z1 = x[t-2]; z2 = x[t-1]; y = x[t]
            s1  += z1
            s2  += z2
            sy  += y
            s11 += z1*z1
            s22 += z2*z2
            s12 += z1*z2
            s1y += z1*y
            s2y += z2*y
            if SE; y2 += y*y; end
        end

        # Build tiny 3x3 normal-equation system G * β = g
        G = Matrix{float(eltype(x))}(undef, 3, 3)
        G[1,1] = n1; G[1,2] = s1;  G[1,3] = s2
        G[2,1] = s1; G[2,2] = s11; G[2,3] = s12
        G[3,1] = s2; G[3,2] = s12; G[3,3] = s22

        g = [sy, s1y, s2y]  # creates a small Vector on the heap (size 3)

        # Solve with Cholesky (SPD by construction if columns are full rank)
        F = cholesky(Symmetric(G))  # tiny; fast and stable
        β = F \ g

        if !SE
            return collect(β)  # ensure a Vector{T}
        else
            dof = n1 - 3
            @assert dof > 0 "Not enough df for standard errors (need n > 3)"
            rss = y2 - dot(β, g)
            σ2  = rss / dof

            # We need diag(inv(G)) without forming inv(G).
            # Solve G x = e_i for i=1..3 and take x[i] as the i-th diagonal element.
            invdiag = similar(β)
            ei = zeros(eltype(β), 3)
            @inbounds for i in 1:3
                fill!(ei, zero(eltype(β))); ei[i] = one(eltype(β))
                xi = F \ ei
                invdiag[i] = xi[i]
            end
            se = sqrt.(σ2 .* invdiag)
            return collect(β), collect(se)
        end
    end
end




ρ̂2, se2 = AR2_reg(true_data, SE=true)


function indirect_inference2(ρ_hat; H=1000, T=200)
    simulated_shocks = randn(H, T)
    k = length(ρ_hat)
    
    function obj(b)
        ρ̃ = zeros(H, k)
        for sim in 1:H
            sim_data = generate_ma1_data(b; ε_shocks = simulated_shocks[sim, :])
            ρ̃[sim, :] = AR2_reg(sim_data) # estimate auxiliary model
        end
        ρ_mean = mean.(eachcol(ρ̃))
        return (ρ_mean - ρ_hat) ⋅ (ρ_mean - ρ_hat)
    end

    result = optimize(obj, -0.99, 0.99)
    return result.minimizer
end

ρ̃2 = indirect_inference2(ρ̂2)


ρ̃_mat2 = zeros(H, 3)
se_mat2 = zeros(H, 3)

for sim in 1:H
    sim_data = generate_ma1_data(b_0 ; ε_shocks = simulated_shocks[sim, :])
    ρ̃, se = AR2_reg(sim_data; SE=true)
    ρ̃_mat2[sim, :] = ρ̃
    se_mat2[sim, :] = se
end
ρ_mean2 = mean.(eachcol(ρ̃_mat2)) # mean at data estimate b̂
se_mean2 = mean.(eachcol(se_mat2)) # mean of standard errors







############### Plots - scale up T ####################
# make plots of ρ estimates across T 
T_vec = collect(100:100:10000)
ρ̂_vec = zeros(length(T_vec), 2)
b̂_vec = zeros(length(T_vec))
b_II_vec = zeros(length(T_vec))

Random.seed!(1234)
for (i, T) in enumerate(T_vec)
    shocks = randn(T)
    true_data = generate_ma1_data(b_0; ε_shocks=shocks)
    ρ̂_vec[i, :] = AR1_reg(true_data)
    b̂_vec[i] = cov(true_data[2:end], true_data[1:end-1])
    b_II_vec[i] = indirect_inference(ρ̂_vec[i, :]; T=T)[1]
end


plot(T_vec, -b̂_vec, label="b̂ from data", xlabel="T", ylabel="b", title="b estimates vs T", linewidth=2)
# change line thickness

plot!(T_vec, -b_II_vec, label="b̂ from II", linewidth=2)

# add horizontal dotted line at b_0
hline!([-b_0], label="b₀", linestyle=:dash)





######################## Monte Carlo for II (AR(1) vs AR(2)) ########################

"""
    mc_indirect_inference(b_true; T_list=[200,500,1000,2000,5000], S=10_000, H=1000, seed=12345)

Runs Monte Carlo experiments for the AR(1)- and AR(2)-based indirect inference estimators of the MA(1) parameter.
For each T in T_list:
  1) Simulate S datasets from the true MA(1) with parameter b_true
  2) For each dataset, compute the AR(1) and AR(2) auxiliary estimates
  3) Run indirect inference using those auxiliary estimates to back out b_II (one scalar)
  4) Collect mean and sampling variance of b_II across the S simulations

Returns a Dict with summary tables (as NamedTuples of Arrays) and also prints a compact report.
"""
function mc_indirect_inference(b_true; T_list=[200,500,1000,2000,5000], S=10_000, H=1000, seed=12345)
    Random.seed!(seed)

    # containers: rows = length(T_list), columns = 2 (AR1, AR2)
    mean_bII = zeros(length(T_list), 2)
    var_bII  = zeros(length(T_list), 2)

    # store full draws if you want (commented by default to save memory)
    # bII_draws_AR1 = Dict{Int, Vector{Float64}}()
    # bII_draws_AR2 = Dict{Int, Vector{Float64}}()

    for (iT, T) in enumerate(T_list)
        # outer shocks reused across AR(1) and AR(2) for comparability
        outer_shocks = randn(S, T)

        bII_AR1 = zeros(S)
        bII_AR2 = zeros(S)

        for s in 1:S
            y = generate_ma1_data(b_true; ε_shocks = outer_shocks[s, :])

            # auxiliary estimates (include intercept as in your AR1_reg/AR2_reg defaults)
            ρ̂1 = AR1_reg(y)                # length 2
            ρ̂2 = AR2_reg(y)                # length 3

            # indirect inference to back out MA(1) parameter (scalar)
            bII_AR1[s] = indirect_inference(ρ̂1; H=H, T=T)[1]
            bII_AR2[s] = indirect_inference2(ρ̂2; H=H, T=T)[1]
        end

        mean_bII[iT, 1] = mean(bII_AR1)
        var_bII[iT,  1] = var(bII_AR1)  # sampling variance

        mean_bII[iT, 2] = mean(bII_AR2)
        var_bII[iT,  2] = var(bII_AR2)

        # bII_draws_AR1[T] = bII_AR1
        # bII_draws_AR2[T] = bII_AR2

        println("T = $T")
        println("  AR(1) II: mean(b) = $(round(mean_bII[iT,1], digits=6)),  var(b) = $(round(var_bII[iT,1], digits=6))")
        println("  AR(2) II: mean(b) = $(round(mean_bII[iT,2], digits=6)),  var(b) = $(round(var_bII[iT,2], digits=6))")
    end

    return Dict(
        :T_list   => T_list,
        :mean_bII => (; AR1 = mean_bII[:,1], AR2 = mean_bII[:,2]),
        :var_bII  => (; AR1 = var_bII[:,1],  AR2 = var_bII[:,2]),
        # :draws_AR1 => bII_draws_AR1,
        # :draws_AR2 => bII_draws_AR2,
    )
end

# Example run
# b_0 is already defined above as the true MA(1) parameter.
# For a quick sanity check, start small (e.g., S=200, H=200) then scale up.
results = mc_indirect_inference(b_0; T_list=[200,500,1000,2000,5000], S=1_000, H=500, seed=1000)
# println(results[:mean_bII])
# println(results[:var_bII])





# make a plot of y = -x / (1+x^2) for x in -1 to 1
x = collect(-1:0.01:1)
y = -x ./ (1 .+ x.^2)
plot(x, y, label="y = -x / (1+x^2)", xlabel="x", ylabel="y", title="Plot of y = -x / (1+x^2)", linewidth=2)