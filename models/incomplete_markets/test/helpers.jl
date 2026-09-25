# Shared setup and economic helpers for the Aiyagari test suite.

const METHODS = ["grid search" => GridSearch, "EGM" => EGM, "interpolated VFI" => InterpVFI]

# Rental rate for the fixed-price tests. It is above δ, so the gross return 1 + r - δ exceeds one, and below the
# complete-markets rate ρ + δ, so households are impatient relative to the return on saving and wealth stays bounded.
const R_FIXED = 0.03

# Silences the solvers' progress messages.
quietly(f) = redirect_stdout(f, devnull)

# The module that defines a solver's Primitives, used to reach its initialize, bellman, etc.
solver(prim) = parentmodule(typeof(prim))

maxdiff(a, b) = maximum(abs.(a .- b))
nanmax(x) = maximum(filter(!isnan, x))


# --- Prices ---

# Capital demand and wage from the Cobb-Douglas firm's first-order conditions at rental rate r.
capital_demand(prim, r) = prim.L * (prim.α / r)^(1 / (1 - prim.α))
wage(prim, r) = (1 - prim.α) * (capital_demand(prim, r) / prim.L)^prim.α

# With complete markets the Euler equation β(1 + r - δ) = 1 pins down the rental rate.
complete_markets_r(prim) = 1 / prim.β - 1 + prim.δ

gross_return(prim, res) = 1 + res.r - prim.δ

# Cash on hand (1 + r - δ)k + w ē z on the policy grid, one column per productivity state.
cash_on_hand(prim, res) = gross_return(prim, res) .* prim.k_grid .+ res.w * prim.ē .* prim.z_grid'

# Long-run distribution of productivity states: the left unit eigenvector of M.
function stationary_z(prim)
    π = nullspace(Matrix(prim.M' - I))[:, 1]
    return π ./ sum(π)
end

function set_prices!(prim, res, r)
    res.r = r
    res.K = capital_demand(prim, r)
    res.w = wage(prim, r)
    return res
end


# --- Household problem ---

solve_household!(prim::GridSearch.Primitives, res) = GridSearch.VFI!(prim, res; tol=1e-10)
solve_household!(prim::EGM.Primitives, res) = EGM.policy_iteration!(prim, res; tol=1e-12)
solve_household!(prim::InterpVFI.Primitives, res) = InterpVFI.VFI!(prim, res; tol=1e-10)

# Solves the household problem at rental rate r, starting from `res` (a fresh initialization by default).
function solve_at_prices(prim, r; res=solver(prim).initialize(prim))
    set_prices!(prim, res, r)
    quietly(() -> solve_household!(prim, res))
    return res
end

# Household solutions at R_FIXED on the default grids, solved once and shared across test files.
const HOUSEHOLD = Dict{Module, Any}()
function baseline_household(m::Module)
    return get!(HOUSEHOLD, m) do
        prim = m.Primitives()
        (prim, solve_at_prices(prim, R_FIXED))
    end
end

# Consumption policy after one more application of the solver's update (Bellman or Euler-equation operator).
updated_consumption(prim::GridSearch.Primitives, res) = GridSearch.bellman(prim, res)[3]
updated_consumption(prim::EGM.Primitives, res) = EGM.bellman(prim, res)[2]
updated_consumption(prim::InterpVFI.Primitives, res) = InterpVFI.bellman(prim, res)[3]

# Sup-norm accuracy of a converged savings policy. Grid search chooses k' from the grid, so its policy is exact.
# Interpolated VFI's policies still move by ~1e-7 once V has met its relative tolerance of 1e-10.
policy_tol(::GridSearch.Primitives) = 0.0
policy_tol(::EGM.Primitives) = 1e-8
policy_tol(::InterpVFI.Primitives) = 1e-5

# Tolerance on k' against the closed form without labor income, and the part of the grid where it applies. Grid search
# rounds k' to its uniform grid. Interpolated VFI is inaccurate just above the constraint, where V ~ log(k) is too
# curved for PCHIP on this grid.
closed_form_tol(prim::GridSearch.Primitives) = (prim.k_grid[2] - prim.k_grid[1], trues(prim.nk))
closed_form_tol(prim::EGM.Primitives) = (1e-6, trues(prim.nk))
closed_form_tol(prim::InterpVFI.Primitives) = (5e-4, prim.k_grid .>= 1)

# Relative Euler-equation errors |ĉ / c - 1|, where ĉ = u'⁻¹(β(1 + r - δ) E[u'(c(k', z'))]) and c(·, z') is a PCHIP
# interpolant of the consumption policy. NaN where the borrowing constraint binds and the Euler equation need not hold.
function euler_errors(prim, res; binding_tol=1e-6)
    (; β, γ, M, nz, k_grid, k_min) = prim
    c_splines = [PchipSpline(k_grid, res.c_policy[:, z]) for z in 1:nz]
    errors = fill(NaN, size(res.c_policy))
    for z in 1:nz
        k_next = res.k_policy[:, z]
        mu_next = reduce(hcat, [evaluate_spline(c_splines[zp], k_next) .^ (-γ) for zp in 1:nz])
        c_euler = (β * gross_return(prim, res) .* (mu_next * M[z, :])) .^ (-1 / γ)
        free = k_next .> k_min + binding_tol
        errors[free, z] = abs.(c_euler[free] ./ res.c_policy[free, z] .- 1)
    end
    return errors
end

# Permanent percentage change in consumption worth a value gap ΔV under log utility.
consumption_equivalent(prim, ΔV) = expm1((1 - prim.β) * ΔV)

# Relative capital-market clearing error in equilibrium. Grid search's capital supply jumps as k' moves between grid
# points, so the root finder only gets it within ~1%.
market_tol(::GridSearch.Primitives) = 1e-2
market_tol(prim) = 2e-3


# --- Distribution ---

# Grid that supports the wealth distribution: the policy grid for grid search, a separate histogram grid otherwise.
dist_grid(prim::GridSearch.Primitives) = prim.k_grid
dist_grid(prim) = prim.k_hist

# Savings policy on the distribution grid.
dist_policy(prim::GridSearch.Primitives, res) = res.k_policy
dist_policy(prim, res) = res.k_pol_hist

# Number of grid points k' is spread over: grid search's k' is a grid point, the others use a two-point lottery.
max_support(::GridSearch.Primitives) = 1
max_support(prim) = 2

# The transition matrix with destination states in rows and source states in columns, so that vec(μ') = P vec(μ).
# Grid search stores the transpose.
forward_operator(prim::GridSearch.Primitives, res) = sparse(transpose(res.T_star))
forward_operator(prim, res) = res.T_star

# Indices of vec(μ) that hold productivity state z.
z_block(prim, z) = (z - 1) * length(dist_grid(prim)) .+ (1:length(dist_grid(prim)))

# Stationary distribution reached from a point mass on (k index i, z), for the policy stored in `res`.
function stationary_from(prim, res, i, z)
    μ0 = zeros(length(dist_grid(prim)), prim.nz)
    μ0[i, z] = 1.0
    res.μ = μ0
    quietly(() -> solver(prim).steady_dist!(prim, res; tol=1e-13, max_iter=200_000))
    return copy(res.μ)
end
