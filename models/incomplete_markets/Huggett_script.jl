using LinearAlgebra, Plots
using Spline # custom spline package

module gridsearch
    using Parameters, LinearAlgebra, Optim, SparseArrays
    include("Huggett_gridsearch.jl")
end


module EGM
    using Parameters, LinearAlgebra, Optim, SparseArrays, Spline
    include("Huggett_EGM.jl")
end


# Moves a pmf between grids through its CDF, so distributions on grids with different spacing are comparable
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


# --------------------- Baseline Comparison ---------------------#

# Both methods use their default grids, which share the same bounds
@time prim_egm, res_egm = EGM.solve_model()

@time prim_gs, res_gs = gridsearch.solve_model()

# equilibrium bond prices and the implied interest rates per model period
res_egm.q, res_gs.q
1 / res_egm.q - 1, 1 / res_gs.q - 1

# bond market clearing: aggregate assets should be zero
res_egm.A, res_gs.A

# wealth inequality and welfare
res_egm.gini, res_gs.gini
res_egm.W, res_gs.W

# plot asset policy functions
plot(prim_gs.a_grid, res_gs.a_policy, label=["GS - employed" "GS - unemployed"], xlabel="a", ylabel="a'", legend=:bottomright)
plot!(prim_egm.a_grid, res_egm.a_policy, linestyle=:dash, label=["EGM - employed" "EGM - unemployed"])
# plot 45 degree line
plot!(prim_gs.a_grid, prim_gs.a_grid, linestyle=:dot, color=:black, label="45° line", title="Asset Policy Function Comparison")


# plot the savings function a' - a
plot(prim_gs.a_grid, res_gs.a_policy .- prim_gs.a_grid, label=["GS - employed" "GS - unemployed"], xlabel="a", ylabel="Savings (a' - a)", legend=:bottomleft)
plot!(prim_egm.a_grid, res_egm.a_policy .- prim_egm.a_grid, linestyle=:dash, label=["EGM - employed" "EGM - unemployed"])
hline!([0.0], linestyle=:dot, color=:black, label="Zero Savings Line", title="Savings Function Comparison")


# plot consumption policy functions
plot(prim_gs.a_grid, res_gs.c_policy, label=["GS - employed" "GS - unemployed"], xlabel="a", ylabel="c", legend=:bottomright)
plot!(prim_egm.a_grid, res_egm.c_policy, linestyle=:dash, label=["EGM - employed" "EGM - unemployed"], title="Consumption Policy Functions")


# plot value functions (EGM recovers V from its policies once the equilibrium is found)
plot(prim_gs.a_grid, res_gs.V, label=["GS - employed" "GS - unemployed"], xlabel="a", ylabel="V(a, s)", legend=:bottomright)
plot!(prim_egm.a_grid, res_egm.V, linestyle=:dash, label=["EGM - employed" "EGM - unemployed"], title="Value Functions")


# plot the marginal pdf of asset holdings. Near their target wealth the employed save less than half a grid step per
# period, which grid search rounds to a' = a: they stop accumulating below the top of EGM's ergodic set and pile up at
# one grid point. The bias barely moves q, and it fades as na grows (by na ≈ 1000 the Gini matches EGM's).
μ_combined_gs = sum(res_gs.μ, dims=2)
# the grid search grid is finer than the EGM histogram grid, so move the EGM distribution onto it
μ_combined_egm = resample_pdf(vec(sum(res_egm.μ, dims=2)), prim_egm.a_hist, prim_gs.a_grid)

plot(prim_gs.a_grid, μ_combined_gs, label="GS", xlabel="a", ylabel="Density", legend=:topright, lw=2)
plot!(prim_gs.a_grid, μ_combined_egm, linestyle=:dash, label="EGM", lw=2)


# plot Lorenz curves of wealth a + s
plot(res_gs.cum_pop, res_gs.lorenz, label="GS", xlabel="Fraction of Agents", ylabel="Cumulative Wealth Share", legend=:topleft, lw=2)
plot!(res_egm.cum_pop, res_egm.lorenz, linestyle=:dash, label="EGM", lw=2)
plot!([0, 1], [0, 1], linestyle=:dot, color=:black, label="Line of Equality", title="Lorenz Curves")
