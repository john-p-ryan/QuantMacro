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
- [objectives.jl](src/objectives.jl): moment-matching criteria, evaluation diagnostics, and model failure types.
- [transforms.jl](src/transforms.jl): maps bounded, unbounded, and fixed parameters to the unit box.
- [local.jl](src/local.jl): local optimizer interfaces, built-in methods, and model evaluation handling.
- [executors.jl](src/executors.jl): serial, threaded, and distributed job execution.
- [storage.jl](src/storage.jl): evaluation caching, budget tracking, journals, snapshots, and restart state.

## Reference

Arnoud, Antoine, Fatih Guvenen, and Tatjana Kleineberg (2022).
[*Benchmarking Global Optimizers*](https://www.fatihguvenen.com/s/AGK2022_v57_FG.pdf).
Working paper, November 23.
