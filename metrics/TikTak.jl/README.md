# TikTak for structural estimation

A self-contained, model-agnostic Julia package for the TikTak global
optimizer. Copy this folder anywhere; it has no dependencies on the rest of
the repository. It accepts a scalar objective or a model that returns moments,
screens scrambled Sobol points, and runs asynchronous derivative-free local
searches on Distributed worker processes, threads, or a single process. Every
model evaluation is journaled before the search proceeds, including moments,
residuals, failure messages, and runtime, so runs can be stopped, resumed, and
reused.

This is a Julia port of the Python package in `../TikTak`. It preserves that
package's interface and semantics (bounds transformations, failure handling,
hard evaluation budgets, restartable runs, warm starts) and replaces its
single-host SQLite design with a coordinator-owned store that works across
nodes.

## Install and run

From this folder, using Julia 1.10 or newer:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
julia --project=. examples/simple_examples.jl --workers 2
```

To use it from another environment, `Pkg.develop(path="path/to/TikTak.jl")`.

The examples fit a quadratic, the multimodal Rastrigin function, an analytic
AR(1) income process, and a staircase objective with infeasible regions.
Output goes to `tiktak-runs/simple/`. Use a different `--output` directory to
run them again. These are small correctness checks, not evidence of
performance on large structural models.

The checked-in `examples/validated_results.json` records a two-worker run and
the test environment. In that run, all 251 tests passed; the example losses
were approximately `6.2e-11` (quadratic), `2.3e-9` (Rastrigin), `7.7e-16`
(AR(1)), and zero (staircase), with 77 safely recorded failures in the
staircase case. Parallel completion order can change evaluation counts between
runs.

## Model interface

```julia
using Distributed
addprocs(4; exeflags="--project=$(Base.active_project())")
@everywhere using TikTak

@everywhere function model_moments(θ)
    ρ, σ = θ
    abs(ρ) >= 1 && throw(ModelEvaluationError("no stationary distribution"))
    variance = σ^2 / (1 - ρ^2)
    return [variance, ρ * variance, ρ^2 * variance]
end

target = model_moments([0.65, 0.2])
criterion = MomentObjective(model_moments, target; scales=target)
result = minimize(criterion, [(0, 0.98), (0, Inf)];
                  scale=[1, 0.2],
                  config=TikTakConfig(n_samples=64, n_local=8, local_max_evals=250, max_evals=3000),
                  run_dir="ar1-run", problem_id="ar1-data-v1-model-v1-draws-v1")
result.x, result.fun, result.moments, result.status
```

Replace `model_moments` with the full equilibrium solution/simulation. With
worker processes, the objective and everything it uses must be defined on
every worker (`@everywhere`), and each worker must have run `using TikTak`.
Anonymous functions are shipped automatically as long as what they reference
exists on the workers. Without worker processes, ordinary closures work. The
objective receives a fresh `Vector{Float64}` of **physical parameters** on
every model call.

The criterion is `r'r`, where `r = L * ((model_moments - target) ./ scales)` and
`L'L = W`. `weights` accepts a nonnegative diagonal vector or a symmetric
positive-semidefinite matrix. Diagonal weighting takes linear time/storage in
the number of moments. Weights default to identity and scales to one. Choose
scales and weights according to the economic/statistical problem: a covariance
matrix itself is not an inverse-covariance weighting matrix. Zero empirical
moments need a positive external scale, rather than automatic division by zero.

Alternatively, pass `objective(θ) -> Real` for a preexisting loss, or return
`Evaluation(value; moments=..., residuals=...)` to preserve your own
diagnostics. This package minimizes; negate a scalar criterion if you need
maximization. It produces point estimates and search diagnostics, not standard
errors or identification tests.

## Parallel execution

Parallelism is across model evaluations and local searches, not inside a
sequential local optimizer. The coordinator (the process calling `minimize`)
owns the run directory and the in-memory run state; workers evaluate the model
and report back through remote calls. Workers therefore need no shared
filesystem, and the run directory can live on the coordinator's local disk.

| Backend | How | Notes |
| --- | --- | --- |
| Distributed (default when workers exist) | `addprocs(n)` before `minimize`; or `workers=[2, 3]` | One job per worker at a time. Memory use includes one model per worker. |
| Serial | `workers=Int[]` or `executor=InlineExecutor()` | Reproducible with a deterministic objective and fixed seed. |
| Threads | `executor=ThreadedExecutor(n)` | The objective must be thread-safe. |
| Custom | `executor=<:AbstractExecutor` | Implement `capacity` and `submit!`; see `src/executors.jl`. |

On a cluster, add workers with the manager for your scheduler (for example
`ClusterManagers.SlurmManager`) and pass the project through `exeflags`. Prevent
nested BLAS or model parallelism from oversubscribing the node, for example
with `BLAS.set_num_threads(1)` on every worker. Prefer a soft deadline
(`max_seconds`) early enough to let running model calls finish before the
scheduler's hard wall time. Each remote store call is a small round trip to the
coordinator, which is negligible for model evaluations that take milliseconds
or more.

Asynchronous completion order can change the search path with multiple
workers. After a hard job termination, ensure all previous workers have exited
before resuming; the coordinator cannot identify orphan workers on your behalf.

## Algorithm and relation to the paper

The implementation follows Arnoud, Guvenen, and Kleineberg (2022), *Benchmarking
Global Optimizers*, Section 2.1 and Appendix A.6:

1. Draw `n_samples` scrambled Sobol points in the transformed box and evaluate
   them in parallel. Powers of two preserve Sobol balance; other sizes take a
   prefix of the next power-of-two draw without making extra model calls.
2. Discard invalid evaluations, rank the finite losses, and retain `n_local`
   seeds (default: 10% of `n_samples`, rounded up). Optional warm starts join
   this screening pool and are reevaluated. If fewer seeds are feasible, use
   all feasible seeds.
3. Run the first local search from the best seed. For subsequent seed `i`
   (one-based), start at `(1 - weight) * seed + weight * incumbent`, with
   `weight = clamp((i / number_of_seeds)^0.5, 0.1, 0.995)` by default. The
   incumbent is the best point reported by completed local searches. Each
   local search returns its best evaluated point even if it hits its budget.
4. Keep as many local searches in flight as the executor has capacity,
   launching a new one as a worker finishes. Newly launched searches use the
   latest completed local results. If a mixed start is infeasible, fall back
   to its feasible seed.
5. Return the best finite point evaluated anywhere, including screening and
   interrupted local searches (full-accuracy evaluations only in a staged run;
   see below). Stop when all seeds have been processed, a
   budget is reached, or the optional `target_value` is attained.

This is an independent implementation of the simplified algorithm, not a port
of the authors' full production code. Extensions here include scrambling,
transformations for infinite bounds, failure handling, journaled persistence,
and asynchronous scheduling. The paper benchmarks Nelder–Mead and DFNLS; our
default is a bounded adaptive Nelder–Mead, and **we do not implement the paper's
DFNLS solver**. See the [authors' TikTak page](https://www.fatihguvenen.com/tiktak)
for their code.

Sobol scrambling is a seed-dependent digital shift: the 32-bit digits of every
coordinate are XORed with a random word per dimension, which permutes the
elementary intervals and so keeps the balance of the unscrambled net.

Mixing occurs in transformed coordinates. For finite bounds this is equivalent
to mixing physical parameters; for infinite bounds the map is nonlinear.
Serial execution with a deterministic objective and fixed seed is reproducible.
No heuristic claiming repeated local agreement proves global optimality is used.

## Choosing a local optimizer

| `local_method` | Use and limitations |
| --- | --- |
| `NelderMeadLocal()` (default) | Bounded adaptive simplex with an explicit full-rank inward initial simplex and trial points clipped to the box; pure Julia, no dependencies. Handles failed (infinite) evaluations. Comparable to the paper's TikTak-nm; simplex collapse at a bound and poor high-dimensional scaling remain possible. |
| `PatternSearchLocal()` | Opportunistic coordinate polling; no fitted surrogate or derivatives. Handles invalid trial points and jumps simply, but can stop on plateaus or miss diagonal descent directions. This is not MADS. |
| `NLoptLocal(:LN_BOBYQA)` | Any bound-constrained derivative-free NLopt algorithm (`:LN_BOBYQA`, `:LN_COBYLA`, `:LN_SBPLX`, `:LN_NELDERMEAD`, ...). Requires `using NLopt` on every process. BOBYQA's quadratic trust-region models are the closest analogue of the Python package's COBYQA default and are often the most efficient choice for costly objectives with some local structure; severe discontinuities, infinite failures, or simulation noise can undermine the surrogate. |
| `CustomLocal(solve; name)` | Your own solver: `solve(f, start, config) -> (converged, message)`, with `f` on the unit box returning `Inf` for failures. Let `TikTak.BudgetExhausted` and `TikTak.LocalBudgetExhausted` propagate. |

`local_max_evals` caps new model attempts per local search. Optimizers also have
a call-count cap of that size (plus the initial feasibility check); cached
calls may make actual model work smaller. `initial_step` and `x_tol` are in
unit-box coordinates. `f_tol` is an absolute improvement threshold for pattern
search and an absolute function tolerance for Nelder–Mead and NLopt. Extra
NLopt settings go through `NLoptLocal(:LN_BOBYQA; xtol_rel=1e-8)`. For
discretized models, set tolerances above the numerical noise floor and compare
backends on a limited pilot budget.

Throw `ModelEvaluationError` for expected failures such as equilibrium
nonexistence or iteration nonconvergence. Nonfinite objectives/moments are also
recorded as failed evaluations and receive an infinite barrier, so failure can
never beat a legitimate large objective. Unexpected exceptions propagate and
are recorded. Add known model exceptions (for example `DomainError` or
`LinearAlgebra.SingularException`) to `failure_exceptions` explicitly; avoid
blanket suppression of programming errors. A disconnected feasible region can
still be missed if screening does not find it. None of these local algorithms
guarantees a solution on an arbitrary discontinuous objective.

## Staged accuracy

Screening only has to rank points, so it can often use a cheaper version of the
model: coarser asset grids, fewer simulated agents, looser inner fixed-point
tolerances. Pass that version as `screen_objective`:

```julia
coarse = MomentObjective(θ -> model_moments(θ; grid=:coarse, tol=1e-6), target; scales=target)
fine   = MomentObjective(θ -> model_moments(θ; grid=:fine,   tol=1e-10), target; scales=target)
result = minimize(fine, bounds; screen_objective=coarse,
                  config=TikTakConfig(n_samples=1024, n_local=20),
                  run_dir="run-v1", problem_id="model-v1-coarse-v1-fine-v1")
```

Sobol and warm-start points are evaluated by `screen_objective`, and those
values are used only to pick and order the seeds. Every local search, including
its starting point, uses `objective`. The incumbent used for mixing,
`target_value`, `result`, and `load_estimates` rank all use full-accuracy values,
because a coarse criterion is biased, not merely noisy, and its values are not
comparable to the fine ones. The two stages are cached separately, so the same
point evaluated by both costs two model calls, and both share `max_evals`
(`local_max_evals` applies only to local searches). The screening objective,
and its moment targets/weights/scales, are part of the run specification. The
`problem_id` should version both accuracy levels.

The coarse and fine criteria should have the same shape: a screening objective
whose ranking of regions differs from the fine one will pick the wrong seeds.
Check this on a small pilot by comparing the two rankings on the same Sobol
points. For more than two accuracy levels, or to change `x_tol` and
`local_max_evals` between stages, chain runs with `warm_start` (see
[Checkpoints, budgets, and reuse](#checkpoints-budgets-and-reuse)). Keep a
single accuracy within each local search: simplex and trust-region methods
compare values across iterations.

## Bounds and scaling

Finite `[a,b]` bounds use `x=(1-u)*a+u*b`. Equal finite bounds fix a parameter and
remove it from the optimization dimension. Physical endpoints remain reachable.
For infinite bounds, with scale `s > 0` and `e=tail` (default `1e-6`):

| Physical bounds | Map from `u` in `[0,1]` |
| --- | --- |
| `[a,+inf)` | `z=(1-e)*u`; `x=a+s*z/(1-z)` |
| `(-inf,b]` | `z=(1-e)*(1-u)`; `x=b-s*z/(1-z)` |
| `(-inf,+inf)` | `z=e+(1-2*e)*u`; `x=location+s*tan(pi*(z-0.5))` |

There is no uniform probability distribution over an infinite range. These maps
induce a particular search distribution and truncate extreme tails for numerical
safety. A smaller `tail` expands the reachable range, often at the cost of
extreme, expensive failed evaluations. Choose `scale` in economically meaningful
units, and `location` near a plausible center for fully unbounded parameters.
Inspect `to_parameters(BoxTransform(bounds; ...), zeros(d))` and its all-ones
counterpart to see the actual computational bounds. Optima outside those
cutoffs cannot be found in that run. If estimates accumulate at a cutoff, revise
the transformation and start a new run with saved estimates as warm starts.
Warm starts outside the cutoff are rejected, not silently clipped.

## Checkpoints, budgets, and reuse

Each run directory contains:

- `history.jsonl`: an append-only journal of the configuration, screening
  points, every distinct evaluated unit/physical vector, moment/residual
  vectors, status/error/runtime, a ledger of model-call attempts, and local
  search starts/results, one JSON record per line. Replaying it reproduces the
  run state exactly; a partial final line left by a crash is ignored.
- `result.json`: a portable best-estimate snapshot, atomically replaced after
  each local search and at return. The journal remains authoritative if the
  process dies before the next snapshot.
- `coordinator.lock`: a pid lock preventing two coordinators from owning the
  same directory. A lock left by a dead process is reclaimed.

```julia
# Same model, target, bounds, simulation draws, and algorithm settings:
result = minimize(criterion, bounds; config=TikTakConfig(config; max_evals=20_000),
                  run_dir="run-v1", problem_id="model-data-draws-v1", resume=true)

# New model/data/settings: reuse coordinates, recompute all objective values.
result = minimize(new_criterion, new_bounds; config=new_config,
                  run_dir="run-v2", problem_id="model-data-draws-v2",
                  warm_start=load_estimates("run-v1"; limit=20))
```

Supply the same transformation arguments when resuming. Only `max_evals`,
`max_seconds`, and the execution backend may change; other settings are
validated. A versioned `problem_id` is required for explicit run directories.
It must change when model code, targets, solver accuracy, input data, or
simulation draws change. Moment targets/weights/scales are additionally
validated automatically. The package cannot infer arbitrary model-code changes
or mutable external inputs.

`max_evals` is a **hard cumulative cap across resumes**, reserved before model
calls; failed calls count. An interrupted attempt retains its charge and its
point may be retried on resume. Identical parameter points share a cached
evaluation, even across workers; nearby points are never rounded together.
This requires deterministic evaluations, usually with fixed simulation draws
(common random numbers). Failed points are cached too. Change `problem_id` and
use a new run if you need to retry transient failures or change simulation
draws.

Completed local searches are skipped on resume. Interrupted local searches
replay from their saved start and use cached evaluations; the internal simplex
or trust-region state is not serialized. Their prior model attempts still count
against their local allocation. `n_evals` counts attempts, while `n_failed`
counts distinct currently failed/error/abandoned points. `status == :completed`
means the allocated searches finished, not that every local solver converged;
inspect `local_results`. `has_solution(result)` means a finite feasible
estimate exists, not global optimality.

`max_seconds` is a **soft per-invocation deadline** checked before new model
calls. Calls already running finish and save their results; an hours-long
evaluation can therefore exceed the deadline by hours. There is no in-process
timeout that can safely kill an arbitrary model. A `callback(result)` runs on
the coordinator after each completed local search for progress logging.

Journal records are flushed to the operating system after every model call and
synced to disk at local-search boundaries and at return, so a process crash
loses nothing and a power loss at most the last few in-flight evaluations. To
move a run to another machine, stop it first and copy the whole run directory.

## Differences from the Python package

- Local solvers: `NelderMeadLocal` (default), `PatternSearchLocal`, `NLoptLocal`
  (BOBYQA and others, replacing SciPy's COBYQA/Powell), and `CustomLocal`.
- Storage: a JSON-lines journal owned by the coordinator instead of SQLite
  shared through the filesystem, so workers can run on other nodes.
- Execution: Julia `Distributed` workers, threads, or inline, selected through
  `workers`/`executor` keywords of `minimize` rather than `TikTakConfig`.
- Sobol scrambling is a digital shift rather than SciPy's Owen scrambling; the
  screening points for a given seed differ from the Python package's.
- Statuses are symbols (`:completed`, ...), `local_results` are `LocalResult`
  structs, and `load_estimates` returns a vector of parameter vectors.
