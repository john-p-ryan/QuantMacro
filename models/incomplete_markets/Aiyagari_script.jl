using Parameters, LinearAlgebra, Optim, SparseArrays, Statistics, Random, Plots
using Spline # custom spline package

module gridsearch
    using Parameters, LinearAlgebra, Optim, SparseArrays
    include("Aiyagari_gridsearch.jl")
end


module EGM
    using Parameters, LinearAlgebra, Optim, SparseArrays, Spline
    include("Aiyagari_EGM.jl")
end


# --------------------- Baseline Comparison ---------------------#

# Baseline values
k_max = 40.0
nk = 800
n_hist = nk  # for EGM


@time prim_egm, res_egm = EGM.solve_model(k_max=k_max, nk=nk, n_hist=n_hist)

@time prim_gs, res_gs = gridsearch.solve_model(k_max=k_max, nk=nk)

# plot capital policy functions
plot(prim_gs.k_grid, res_gs.k_policy, label=["GS - ϵ high" "GS - ϵ low"], xlabel="k", ylabel="k'", legend=:bottomright)
plot!(prim_egm.k_grid, res_egm.k_policy, linestyle=:dash, label=["EGM - ϵ high" "EGM - ϵ low"])
# plot 45 degree line
plot!(prim_gs.k_grid, prim_gs.k_grid, linestyle=:dot, color=:black, label="45° line", title="Capital Policy Function Comparison")


# plot the savings function k' - k
plot(prim_gs.k_grid, res_gs.k_policy[:, 1] .- prim_gs.k_grid, label="GS - ϵ high", xlabel="k", ylabel="Savings (k' - k)", legend=:bottomright)
plot!(prim_gs.k_grid, res_gs.k_policy[:, 2] .- prim_gs.k_grid, label="GS - ϵ low")
plot!(prim_egm.k_grid, res_egm.k_policy[:, 1] .- prim_egm.k_grid, linestyle=:dash, label="EGM - ϵ high")
plot!(prim_egm.k_grid, res_egm.k_policy[:, 2] .- prim_egm.k_grid, linestyle=:dash, label="EGM - ϵ low")
#hline!(0.0, linestyle=:dot, color=:black, label="Zero Savings Line", title="Savings Function Comparison")



# plot consumption policy functions
plot(prim_gs.k_grid, res_gs.c_policy, label=["GS - ϵ high" "GS - ϵ low"], xlabel="k", ylabel="c", legend=:bottomright)
plot!(prim_egm.k_grid, res_egm.c_policy, linestyle=:dash, label=["EGM - ϵ high" "EGM - ϵ low"], title="Consumption Policy Functions")


# plot the marginal pdf of capital holdings
μ_combined_gs = sum(res_gs.μ, dims=2)
μ_combined_egm = sum(res_egm.μ, dims=2)


plot(prim_gs.k_grid, μ_combined_gs, label="GS", xlabel="k", ylabel="Density", legend=:topright, lw=2)
plot!(prim_gs.k_grid, μ_combined_egm, linestyle=:dash, label="EGM", lw=2)






# ------------------------ Compare grid search with finer grids to EGM coarse grid ------------------------#

k_max = 100.0
nk_fine = 20000
n_hist = 2000
nk_egm = 1000

@time prim_egm_coarse, res_egm_coarse = EGM.solve_model(k_max=k_max, nk=nk_egm, n_hist=n_hist)

@time prim_gs_fine, res_gs_fine = gridsearch.solve_model(k_max=k_max, nk=nk_fine)


# plot capital policy functions
plot(prim_gs_fine.k_grid, res_gs_fine.k_policy, label=["GS - ϵ high" "GS - ϵ low"], xlabel="k", ylabel="k'", legend=:bottomright)
plot!(prim_egm_coarse.k_grid, res_egm_coarse.k_policy, linestyle=:dash, label=["EGM - ϵ high" "EGM - ϵ low"])
# plot 45 degree line
plot!(prim_gs_fine.k_grid, prim_gs_fine.k_grid, linestyle=:dot, color=:black, label="45° line", title="Capital Policy Function Comparison (Fine Grid vs Coarse Grid)")

# plot the savings function k' - k
plot(prim_gs_fine.k_grid, res_gs_fine.k_policy[:, 1] .- prim_gs_fine.k_grid, label="GS - ϵ high", xlabel="k", ylabel="Savings (k' - k)", legend=:topright)
plot!(prim_gs_fine.k_grid, res_gs_fine.k_policy[:, 2] .- prim_gs_fine.k_grid, label="GS - ϵ low")
plot!(prim_egm_coarse.k_grid, res_egm_coarse.k_policy[:, 1] .- prim_egm_coarse.k_grid, linestyle=:dash, label="EGM - ϵ high")
plot!(prim_egm_coarse.k_grid, res_egm_coarse.k_policy[:, 2] .- prim_egm_coarse.k_grid, linestyle=:dash, label="EGM - ϵ low")
#hline!(0.0, linestyle=:dot, color=:black, label="Zero Savings Line", title="Savings Function Comparison")

# plot consumption policy functions
plot(prim_gs_fine.k_grid, res_gs_fine.c_policy, label=["GS - ϵ high" "GS - ϵ low"], xlabel="k", ylabel="c", legend=:bottomright)
plot!(prim_egm_coarse.k_grid, res_egm_coarse.c_policy, linestyle=:dash, label=["EGM - ϵ high" "EGM - ϵ low"])

# plot the marginal pdf of capital holdings
μ_combined_gs_fine = sum(res_gs_fine.μ, dims=2)
μ_combined_egm_coarse = sum(res_egm_coarse.μ, dims=2)
# correct for different grid spacing
function resample_pdf(pdf_original::AbstractVector{<:Real},
                      grid_original::AbstractVector{<:Real},
                      grid_new::AbstractVector{<:Real})

    # 1. Calculate the cumulative distribution function (CDF) on the original grid.
    #    Ensure the input pdf is normalized first.
    cdf_original = cumsum(vec(pdf_original) ./ sum(pdf_original))

    # 2. Create a PCHIP spline interpolation of the CDF.
    cdf_spline = Spline.PchipSpline(grid_original, cdf_original)

    # 3. Evaluate the CDF spline on the new grid points.
    cdf_new = Spline.evaluate_spline(cdf_spline, grid_new)
    
    # Clamp CDF values between 0 and 1 to handle potential minor extrapolation errors.
    clamp!(cdf_new, 0.0, 1.0)

    # 4. Differentiate the new CDF to get the new PDF.
    #    The probability at grid[i] is cdf[i] - cdf[i-1].
    pdf_new = similar(cdf_new)
    pdf_new[1] = cdf_new[1] # Mass at the first point
    @inbounds for i in 2:length(cdf_new)
        pdf_new[i] = cdf_new[i] - cdf_new[i-1]
    end

    # Guard against tiny negative probabilities from floating point inaccuracies.
    pdf_new = max.(0.0, pdf_new)

    # 5. Normalize the final PDF to ensure it's a valid probability distribution.
    if sum(pdf_new) > 1e-9
        pdf_new ./= sum(pdf_new)
    end

    return pdf_new
end
μ_combined_egm_coarse = resample_pdf(vec(μ_combined_egm_coarse), prim_egm_coarse.k_hist, prim_gs_fine.k_grid)
plot(prim_gs_fine.k_grid, μ_combined_gs_fine, label="GS", xlabel="k", ylabel="Density", legend=:topright, lw=2)
plot!(prim_gs_fine.k_grid, μ_combined_egm_coarse, linestyle=:dash, label="EGM", lw=2)