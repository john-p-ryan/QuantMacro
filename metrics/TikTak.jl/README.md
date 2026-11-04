# TikTak.jl

A model-agnostic Julia package for global optimization and structural estimation.
The algorithm is based on the TikTak method described by
[Arnoud, Guvenen, and Kleineberg (2022)](#reference).
TikTak screens scrambled Sobol points, then combines promising starting points
with the best completed local solutions to launch derivative-free local searches.
It accepts scalar losses or models that return moments, supports serial and
parallel execution, and saves evaluations so runs can be resumed or reused.

## Quick start

Requires **Julia 1.10 or newer**. From this folder, install dependencies:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Start Julia with `julia --project=.` and minimize a scalar loss:

```julia
using TikTak

loss(θ) = sum((θ .- [0.2, -0.4]) .^ 2)
bounds = [(-2.0, 2.0), (-2.0, 2.0)]
config = TikTakConfig(n_samples=64, n_local=8, max_evals=3000)

result = minimize(loss, bounds; config=config,
                  run_dir="quadratic-run", problem_id="quadratic-v1")

result.x       # Best parameters: approximately [0.2, -0.4]
result.fun     # Best loss: approximately zero
result.status  # Why the search stopped
```

Use a fresh run directory for a new run, or resume an existing one as shown below.
Omitting `run_dir` saves to a temporary directory, available as `result.run_dir`.

To use the package from your own Julia project, activate that project's environment
and register the local package folder:

```julia
using Pkg
Pkg.develop(path="/path/to/TikTak.jl")
using TikTak
```

## Use your own model files

Load your model with `include`. A scalar objective takes a parameter vector and
returns a number to minimize. For moment matching, wrap a function that returns
a vector of model moments:

```julia
include("model.jl")  # Defines model_moments(θ)

target = [1.0, 0.5]  # Replace with your empirical moments
criterion = MomentObjective(model_moments, target; scales=[1.0, 0.5])
result = minimize(criterion, [(0.0, 0.98), (0.0, Inf)];
                  scale=[1.0, 0.2], config=config,
                  run_dir="model-run", problem_id="model-data-draws-v1")

result.moments
result.residuals
```

`MomentObjective` minimizes the weighted squared errors after dividing by
positive `scales`. Optional `weights` accepts a nonnegative diagonal vector or
a symmetric positive-semidefinite matrix; defaults are unit scales and identity
weights. The model always receives parameters in their original units.

Use fixed simulation draws: repeated parameter vectors reuse cached evaluations.
Throw `ModelEvaluationError("reason")` for expected model failures, such as a
missing equilibrium. These and nonfinite values receive an infinite loss;
unexpected exceptions propagate. To save diagnostics with your own scalar loss,
return `Evaluation(value; moments=..., residuals=...)`.

## Common options

Pass search settings through `TikTakConfig(...)`:

| Option | Purpose |
| --- | --- |
| `n_samples`, `n_local` | Screening points and number of local searches. Prefer powers of two for `n_samples`. |
| `max_evals`, `local_max_evals` | Total model-call budget and budget per local search. Failed calls count. |
| `max_seconds`, `target_value` | Soft time limit or stop at a desired loss. Running calls finish. |
| `seed` | Screening seed; serial runs are reproducible with a deterministic objective. |
| `initial_step`, `x_tol`, `f_tol` | Local-search step and tolerances. Step and parameter tolerance use unit-box coordinates. |
| `local_method` | `NelderMeadLocal()` (default), `PatternSearchLocal()`, `NLoptLocal(:LN_BOBYQA)`, or `CustomLocal(solve; name="my-solver")`. NLopt requires installation and `using NLopt`. |

Additional keywords go directly to `minimize`:

- **Bounds:** finite, infinite, or equal bounds (fixed parameters). Use `scale`
  for infinite bounds, `location` to center fully unbounded parameters, and
  `tail` to control tail truncation. Infinite ranges have finite computational
  cutoffs, so choose scales in meaningful parameter units.
- **Cheaper screening:** `screen_objective=coarse_loss` ranks starting points;
  local searches and the reported result use the main objective.
- **Progress:** `callback=r -> println(r.fun)` runs after each completed local search.
- **Workspace history:** `store_evaluations=true` retains parameter points and
  their evaluations in `result.evaluations` (including callback snapshots).

## Parallel execution

With no workers, execution is serial. To use worker processes, add them **before**
loading your model and define its functions and dependencies on every worker:

```julia
using Distributed
addprocs(4; exeflags="--project=$(Base.active_project())")
@everywhere using TikTak
@everywhere include("/absolute/path/to/model.jl")

# Calls to minimize now use the workers by default.
```

Each worker must be able to read your model and input files. Only the coordinator
needs access to the run directory. Alternatively, pass `workers=Int[]` to force
serial execution or `executor=ThreadedExecutor(4)` for a thread-safe model
(start Julia with `--threads=4`). Parallel completion order can change the search path.

## Resume or reuse a run

Each run saves an evaluation journal (`history.jsonl`) and best-estimate snapshot
(`result.json`). Resume the quick-start run with a larger cumulative budget:

```julia
result = minimize(loss, bounds; config=TikTakConfig(config; max_evals=6000),
                  run_dir="quadratic-run", problem_id="quadratic-v1", resume=true)
```

Stop previous workers before resuming. Keep the objective, data, simulation draws,
bounds, transformations, and search settings unchanged; only `max_evals`,
`max_seconds`, and the execution backend may change. Version `problem_id` whenever
the model or inputs change, and use a new run directory. Reuse saved parameters
as starting points with `warm_start=load_estimates("quadratic-run"; limit=20)`;
their objective values are recomputed in the new run.

`has_solution(result)` indicates a finite feasible estimate. `:completed` means
the allocated searches finished; it does not guarantee global optimality.

## Parameter points and moment evaluations

Keep the evaluation history in your Julia workspace when the search returns:

```julia
result = minimize(criterion, bounds; config, store_evaluations=true)
evaluations = result.evaluations

evaluations[1].parameters  # Physical parameters passed to the model
evaluations[1].moments     # Model moments at that same point
evaluations[1].residuals   # Scaled, weighted residuals for MomentObjective
evaluations[1].value       # Objective value
evaluations[1].status      # :ok, :failed, :error, :abandoned, or :pending
```

Retrieve evaluations later without rerunning the model or parsing JSON:

```julia
evaluations = load_evaluations(result)          # Workspace snapshot, or disk if not retained
evaluations = load_evaluations("model-run")     # Latest saved state, including resumes
good = load_evaluations(result; status=:ok, stage=:full)

# For a nonempty set of successful moment evaluations, one evaluation per row:
parameters = stack([r.parameters for r in good]; dims=1)
moments = stack([r.moments for r in good]; dims=1)
losses = [r.value for r in good]

failures = load_evaluations(result; status=:failed)
[(r.parameters, r.error) for r in failures]
```

Records stay paired and ordered by model-call attempt, rather than completion
time. Repeated cache hits appear once; an abandoned point that is retried keeps
its latest attempt. `screening=true` marks the cheaper objective of a staged
run; use `stage=:full` or `stage=:screening` to analyze the two criteria separately.
A scalar objective has no moments unless it returns `Evaluation(...; moments=...)`;
failed evaluations may also have `moments === nothing`. Only stack moment vectors
of equal length.

`store_evaluations` defaults to `false` to avoid retaining an extra copy of the
cache after the search. When enabled, it includes prior evaluations on resume
and keeps an independent snapshot in the returned result, without creating
global variables. Callback snapshots can include pending calls from other
workers. The journal is always saved, and `result.json` remains a compact
best-estimate snapshot. If a run throws an error, retrieve its recorded
evaluations with `load_evaluations(run_dir)` using an explicit run directory.

## Diagnostics and inference

Plot every moment against every parameter using saved evaluations, without
rerunning the model. Install Plots in your environment (`Pkg.add("Plots")`), then:

```julia
using Plots

panel = binscatter(result; nbins=20,
                   parameter_names=["rho", "sigma"], moment_names=["var", "cov"])
display(panel)
savefig(panel, "moment-binscatter.png")
```

The panel has **moments as rows and parameters as columns**, like the Jacobian;
each cell plots mean raw moments against mean physical parameters in approximately
equal-frequency bins. Tied parameter values stay together, so some columns can
have fewer bins. Names cover all original parameters and moments. Select columns
with `parameters=[2, 1]`, or `parameters=jac.free` to match a Jacobian that omits
fixed parameters. Extra plot keywords such as `size=(1000, 800)` and `markersize=3`
customize the figure. Plots is optional and only needed to render the panel.

`binscatter` also accepts a run directory or a vector from `load_evaluations`.
It uses successful finite moment evaluations from `stage=:full` by default;
pass `stage=:screening` to examine the cheaper objective separately. Scalar
evaluations without moments and failed evaluations are skipped. For the bin
means and counts without a plotting dependency, use `binscatter_data` with the
same options; its `panels[i, j]` contains `x`, `y`, and `count` vectors.

These plots show marginal associations in the search history: other parameters
vary within each bin, and local searches concentrate evaluations near minima.
Their slopes therefore need not equal Jacobian derivatives holding other
parameters fixed.

After a moment-matching search, check the search, the fit, and local
identification before interpreting standard errors:

```julia
summary = search_summary(result; moment_names=["var", "cov"])
summary.minima[1].fit             # Targets, model moments, errors, loss contributions

jac = moment_jacobian(criterion, result.x, bounds; names=["rho", "sigma"])
identification(jac, criterion)    # Singular values and weak parameter combinations
sensitivity = sensitivity_summary(jac, criterion; moment_names=["var", "cov"],
                                  parameter_scales=[0.1, 0.01])
display(sensitivity)             # Named moment responses and target sensitivities
estimates = parameter_covariance(jac, criterion; covariance=Σ, nobs=N, n_simulations=R)
wald_intervals(estimates)         # Throws if rank deficient or near a bound
derived = delta_method(estimates, θ -> θ[2]^2 / (1 - θ[1]^2); names=["variance"])
```

- `search_summary` groups completed local searches into distinct minima and
  reports how often each was found, solver convergence, and distance to bounds.
  A best minimum found only once suggests a poor search; large errors at every
  minimum suggest poor fit.
- `moment_jacobian` takes derivatives through `jacobian=` or uses finite
  differences in original parameter units, one-sided at bounds. It checks that
  the moments are reproducible and reports how the Jacobian changes when the
  step is halved and doubled. With simulated moments, choose `step` large
  relative to any discreteness. Pass `mapper=pmap` to use workers.
- `identification` scales each parameter's column to unit norm by default; pass
  `parameter_scales=:raw` or a vector of parameter units for other normalizations.
- `parameter_covariance` computes the sandwich covariance consistently with the
  objective's `scales` and `weights`. `covariance` is the asymptotic covariance Σ
  of √N times the raw empirical moment errors, so Σ/`nobs` is their covariance.
  Simulation noise in the model moments is a separate term; see below.
- `delta_method` propagates the parameter covariance to derived quantities.

These are local diagnostics: they do not establish global identification.
Criterion profiles, below, look along one parameter beyond the linearization.

### Sampling and simulation noise

With fixed simulation draws independent of the data, the difference between
empirical and simulated moments has covariance `Ω = Σ/N + Σₛ/Nₛ`, and the sandwich
splits accordingly into `estimates.sampling_covariance` and
`estimates.simulation_covariance`; `display(estimates)` shows the simulation share of
each variance.

| Keywords | Simulation term |
| --- | --- |
| none | No separate term: the moments are computed without simulation noise (for example from a stationary distribution on a grid), or Σ already includes it |
| `n_simulations=R` | `Σ/(R N)`: moments averaged over R simulated samples with the size and design of the data, so `Ω = (1 + 1/R) Σ/N` |
| `simulation_covariance=Σₛ, simulation_nobs=Nₛ` | `Σₛ/Nₛ`, for a simulation whose design differs from the data, such as a large stationary population |

`inverse_covariance_weights` and `two_step` take the same keywords, so efficient
weights and the J statistic use the total `Ω`.

### Derived quantities

`delta_method(estimates, g)` gives standard errors for `g(θ)`, such as elasticities,
steady-state objects, or counterfactual changes. `g` takes the full parameter vector,
including fixed parameters, and returns a number or vector; it may solve the model,
with fixed draws. Derivatives use the finite differences of `moment_jacobian`, within
the parameter bounds, with the same `step`, `check_factors`, and `mapper` options, or
pass `jacobian=`. The result keeps the sampling and simulation split, and
`wald_intervals(derived)` refuses it when the parameter covariance is not regular or
the derivatives of `g` are unstable across step sizes.

### Criterion profiles

`profile` fixes one parameter on a grid and reoptimizes the others, each grid point in
its own saved run, warm-started from the estimate and from the neighbouring grid point:

```julia
estimate = two_step(criterion, bounds; covariance=Σ, nobs=N, n_simulations=R, ...)
se = parameter_covariance(jac, estimate).standard_errors
grid = estimate.x[1] .+ se[1] .* (-3:0.5:3)
p = profile(estimate.objective, bounds, estimate.x, "rho", grid; names=["rho", "sigma"],
            config=TikTakConfig(n_samples=32, n_local=3),
            run_dir="model-profiles", problem_id="model-data-draws-v1")
display(p)                  # Loss and increase over the estimate at each grid value
profile_interval(p)         # (lower, upper, lower_bounded, upper_bounded, disjoint)
```

- A flat profile shows weak identification along that parameter, and an asymmetric one
  shows where the normal approximation fails. A grid point with a lower loss than the
  estimate is reported in `p.issues`: the estimation search missed the global minimum.
- `profile_interval` inverts the distance-metric test (Newey and West, 1987): the set of
  values whose loss increase is at most the χ²₁ critical value (3.84 for 95%). The
  limit is χ²₁ only for the efficient criterion on the J-statistic scale, such as the
  second step of `two_step` or unregularized `inverse_covariance_weights`, and only away
  from bounds. Unlike Wald intervals, it is invariant to reparametrization and can be
  asymmetric.
- Grid point `v` of parameter `j` is saved in `run_dir/p<j>_<v>`. With `resume=true`,
  saved points resume and new ones start, so the grid can be refined later. A changed
  loss at the same estimate, or changed bounds, settings, targets, or weights, make the
  resume fail; other model or data changes need a new `problem_id`.
- Each search drops `target_value`, so it reaches the conditional minimum. Each search
  uses the workers; grid points run in sequence.

`sensitivity_summary` reports the raw moment Jacobian and the linearized response
of fitted parameters to empirical moment targets, holding weights and scales
fixed. If `J` is the raw Jacobian, `D = Diagonal(criterion.scales)`, and
`G = D⁻¹J`, the response is `K = (G'WG)⁻¹G'WD⁻¹`. A positive `K[k,i]` means
that increasing target moment `i` raises free parameter `k`; an additive change
in the model moment has the opposite sign. Full weights retain all cross-moment
effects, and entries always refer to the original named moments.

```julia
sensitivity.jacobian                  # moments × free parameters, raw units
sensitivity.scaled_jacobian           # D⁻¹ J C
sensitivity.target_sensitivity        # K: free parameters × moments, raw units
sensitivity.scaled_target_sensitivity # C⁻¹ K D
Δθ = sensitivity.target_sensitivity * Δtarget
```

Here `C = Diagonal(sensitivity.parameter_scales)`. The default
`parameter_scales=:raw` uses one physical unit; supply a positive scalar or vector
over all parameters for economically meaningful changes, or `:column` for the
normalization used by `identification`. Fixed parameters are omitted and
`sensitivity.free` maps rows of `K` back to the original parameters. Reporting
scales do not change raw `K` or rank. To compare weighting choices, call
`sensitivity_summary` with the same Jacobian and objectives carrying the
alternative weights; no model calls or saved-run changes are made.

The calculation uses an SVD with column normalization. Rank deficiency produces
`NaN` target sensitivities; binding bounds and unstable derivatives set
`regular=false` and populate `issues`. At bounds, any reported `K` is the
unconstrained response. Without a moment covariance, proximity to bounds in
standard-error units is not assessed. This local Gauss–Newton approximation
gives the exact target derivative for an interior linear model or a nonlinear
model at an exact fit. With nonzero nonlinear fitting errors, curvature can
change the derivative after reoptimization. It does not establish convergence,
goodness of fit, or the sensitivity to changing weights themselves.

## Weighting matrices and two-step estimation

`inverse_covariance_weights` builds a weighting matrix from the moment covariance,
with the same conventions as `parameter_covariance`: Σ is the asymptotic covariance
of √N times the raw empirical moment errors, and `n_simulations` or
`simulation_covariance` with `simulation_nobs` add simulation noise. Pass the result
as `weights` to any `MomentObjective`:

```julia
w = inverse_covariance_weights(Σ; nobs=N, moment_names=["var", "cov", "cov2"])
display(w)                                         # Correlation eigenvalues and weak moment combinations
efficient = MomentObjective(criterion; weights=w)  # Same model, targets, and scales
```

The objective converts `w` to its own scale units, `W = D Ω⁻¹ D` with `Ω = Σ/N`
and `D = Diagonal(scales)`, so the criterion is `(m - target)'Ω⁻¹(m - target)`
whatever the scales. Its minimum is Hansen's J statistic.

Estimated covariances of many moments are often ill-conditioned, and inverting
them puts large weight on noisy moment combinations, which biases estimates in
finite samples (Altonji and Segal, 1996). Regularization shrinks the moment
correlations towards zero, `Ω_λ = (1-λ)Ω + λ Diagonal(Ω)`, so it does not depend
on moment units or scales:

| Option | Weights |
| --- | --- |
| default | Inverse covariance; throws if the correlation matrix is numerically singular |
| `shrinkage=λ` | Fixed shrinkage, `0 ≤ λ ≤ 1` |
| `max_condition=κ` | The smallest λ for which the correlation matrix has condition number at most κ |
| `diagonal=true` | Inverse variances, stored as a vector so `fit_summary` attributes the loss to each moment |

The report gives the condition number, numerical rank, and eigenvalues of the
correlation matrix, each with its combination of standardized moments; the error
for a singular matrix names the redundant combination. With the same Σ and `nobs`,
`parameter_covariance` gives valid sandwich standard errors under any of these
weights, and `(G'Ω⁻¹G)⁻¹` without regularization.

`two_step` minimizes the objective with its own weights, then runs a second,
separately saved search with inverse-covariance weights. The second search's
starting pool includes the first-step estimate and other first-step minima:

```julia
estimate = two_step(criterion, bounds; covariance=Σ, nobs=N, n_simulations=R,
                    weight_options=(; max_condition=1e4), config=config,
                    run_dir="model-two-step", problem_id="model-data-draws-v1")
estimate.first.x             # First-step estimate
estimate.x                   # Second-step estimate
estimate.j_statistic         # NaN when the weights are regularized
estimate.degrees_of_freedom  # Moments minus free parameters

jac = moment_jacobian(estimate.objective, estimate.x, bounds)
parameter_covariance(jac, estimate)  # Second-step objective and moment covariance
```

- `covariance` and `simulation_covariance` can also be functions `θ -> Σ`, evaluated
  once at the first-step estimate, for example from moment contributions simulated
  there. Keep their draws fixed.
- The steps are saved in `run_dir/step1` and `run_dir/step2-<tag>`, where the tag
  hashes the second-step weights. `resume=true` resumes both. If the weights change,
  a new second-step run starts rather than reusing losses from another criterion;
  only parameter vectors pass between steps.
- `second_config` defaults to `config` without `target_value`. Second-step losses are
  on the scale of the J statistic, so adjust `f_tol` and any target. A smaller
  `n_samples` and `n_local` gives a cheaper refinement.
- If the first step stops early, `estimate.second === nothing`; increase its budget and resume.
- A `screen_objective` must be a `MomentObjective`, which is reweighted for the second step.

## Examples and tests

```sh
julia --project=. examples/simple_examples.jl --workers 2 --output tiktak-runs/demo
julia --project=. -e 'using Pkg; Pkg.test()'
```

[Simple examples](examples/simple_examples.jl) cover a quadratic, Rastrigin,
AR(1) moment matching, and an objective with infeasible regions.
[AR(1) estimation](examples/estimate_ar1.jl) provides a command-line entry point
with worker, budget, and resume options.

## Source files

- [TikTak.jl](src/TikTak.jl): package entry point, exports, and module assembly.
- [solver.jl](src/solver.jl): search configuration, Sobol screening, multistart scheduling, and results.
- [summaries.jl](src/summaries.jl): moment fit decompositions and distinct local minima of a saved run.
- [binscatter.jl](src/binscatter.jl): moment-by-parameter bin means and optional Plots.jl panels.
- [inference.jl](src/inference.jl): moment Jacobians, local identification, sandwich standard errors with sampling and simulation parts, and the delta method.
- [sensitivity.jl](src/sensitivity.jl): named local moment responses and weighted moment-to-parameter sensitivities.
- [weights.jl](src/weights.jl): inverse-covariance weighting matrices, regularization, and conditioning diagnostics.
- [two_step.jl](src/two_step.jl): two-step estimation with versioned, warm-started second-step runs.
- [profile.jl](src/profile.jl): criterion profiles in saved, warm-started runs and profile confidence intervals.
- [objectives.jl](src/objectives.jl): moment-matching criteria, evaluation diagnostics, and model failure types.
- [transforms.jl](src/transforms.jl): maps bounded, unbounded, and fixed parameters to the unit box.
- [local.jl](src/local.jl): local optimizer interfaces, built-in methods, and model evaluation handling.
- [executors.jl](src/executors.jl): serial, threaded, and distributed job execution.
- [storage.jl](src/storage.jl): evaluation caching, budget tracking, journals, snapshots, and restart state.

## Reference

Arnoud, Antoine, Fatih Guvenen, and Tatjana Kleineberg (2022).
[*Benchmarking Global Optimizers*](https://www.fatihguvenen.com/s/AGK2022_v57_FG.pdf).
Working paper, November 23.
