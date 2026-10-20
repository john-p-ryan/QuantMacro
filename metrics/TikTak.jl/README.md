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
estimates = parameter_covariance(jac, criterion; covariance=Σ, nobs=N)
wald_intervals(estimates)         # Throws if rank deficient or near a bound
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
  of √N times the raw moment errors, so Σ/`nobs` is their covariance; include
  simulation noise in Σ.

These are local diagnostics: they do not establish global identification.

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
- [inference.jl](src/inference.jl): moment Jacobians, local identification, and sandwich standard errors.
- [objectives.jl](src/objectives.jl): moment-matching criteria, evaluation diagnostics, and model failure types.
- [transforms.jl](src/transforms.jl): maps bounded, unbounded, and fixed parameters to the unit box.
- [local.jl](src/local.jl): local optimizer interfaces, built-in methods, and model evaluation handling.
- [executors.jl](src/executors.jl): serial, threaded, and distributed job execution.
- [storage.jl](src/storage.jl): evaluation caching, budget tracking, journals, snapshots, and restart state.

## Reference

Arnoud, Antoine, Fatih Guvenen, and Tatjana Kleineberg (2022).
[*Benchmarking Global Optimizers*](https://www.fatihguvenen.com/s/AGK2022_v57_FG.pdf).
Working paper, November 23.
