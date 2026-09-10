# TikTak for structural estimation

A self-contained, installable, model-agnostic Python package. Copy this folder
anywhere; it has no dependencies on the rest of the repository. It accepts a
scalar objective or a model that returns moments, screens Sobol points, and runs
parallel derivative-free local searches. Every model evaluation is saved before
the search proceeds, including moments, residuals, failure messages, and runtime.

## Install and run

From this folder, using Python 3.10 or newer on Linux/macOS:

```sh
python -m pip install -e '.[test]'
python -m pytest -q
python -m examples.simple_examples --workers 2
```

The examples fit a quadratic, the multimodal Rastrigin function, an analytic
AR(1) income process, and a staircase objective with infeasible regions. Output
goes to `tiktak-runs/simple/`. Use a different `--output` directory to run
them again. These are small correctness checks, not evidence of performance on
large structural models.

The checked-in `examples/validated_results.json` records a two-worker run and
the test environment. In that run, all 27 tests passed; the example losses were
approximately `9.6e-31` (quadratic), `1.2e-9` (Rastrigin), `1.7e-18` (AR(1)),
and zero (staircase), with 75 safely recorded failures in the staircase case.
Parallel completion order can change evaluation counts between runs.

## Model interface

```python
import numpy as np
from tiktak import ModelEvaluationError, MomentObjective, TikTakConfig, minimize

def model_moments(theta):
    rho, sigma = theta
    if abs(rho) >= 1:
        raise ModelEvaluationError("no stationary distribution")
    variance = sigma**2 / (1 - rho**2)
    return np.array([variance, rho * variance, rho**2 * variance])

def main():
    target = model_moments([0.65, 0.2])
    criterion = MomentObjective(model_moments, target, scales=target)
    result = minimize(
        criterion,
        bounds=[(0, 0.98), (0, np.inf)],
        scale=[1, 0.2],
        config=TikTakConfig(n_samples=64, n_local=8, workers=4,
                            local_max_evals=250, max_evals=3000),
        run_dir="ar1-run", problem_id="ar1-data-v1-model-v1-draws-v1",
    )
    print(result.x, result.fun, result.moments, result.status)

if __name__ == "__main__":
    main()
```

Replace `model_moments` with the full equilibrium solution/simulation. Model
functions must be importable top-level functions for process execution. Avoid
lambdas and notebook-local functions with `workers > 1`; use a Python module.
With `workers=1`, ordinary closures work. The objective receives a fresh 1D array
of **physical parameters** on every model call.

The criterion is `r'r`, where `r = L @ ((model_moments - target) / scales)` and
`L.T @ L = W`. `weights` accepts a nonnegative diagonal vector or a symmetric
positive-semidefinite matrix. Diagonal weighting takes linear time/storage in
the number of moments. Weights default to identity and scales to one. Choose
scales and weights according to the economic/statistical problem: a covariance
matrix itself is not an inverse-covariance weighting matrix. Zero empirical
moments need a positive external scale, rather than automatic division by zero.

Alternatively, pass `objective(theta) -> float` for a preexisting loss, or return
`Evaluation(value, moments=..., residuals=...)` to preserve your own diagnostics.
This package minimizes; negate a scalar criterion if you need maximization.
It produces point estimates and search diagnostics, not standard errors or
identification tests.

## Algorithm and relation to the paper

The implementation follows Arnoud, Guvenen, and Kleineberg (2022), *Benchmarking
Global Optimizers*, Section 2.1 and Appendix A.6 in the supplied paper:

1. Draw `n_samples` scrambled Sobol points in the transformed box and evaluate
   them in parallel. Powers of two preserve Sobol balance; other sizes take a
   prefix of the next power-of-two draw without making extra model calls.
2. Discard invalid evaluations, rank the finite losses, and retain `n_local`
   seeds (default: 10% of `n_samples`, rounded up). Optional warm starts join
   this screening pool and are reevaluated. If fewer seeds are feasible, use
   all feasible seeds.
3. Run the first local search from the best seed. For subsequent seed `i`
   (one-based), start at `(1 - weight) * seed + weight * incumbent`, with
   `weight = clip((i / number_of_seeds)**0.5, 0.1, 0.995)` by default. The
   incumbent is the best point reported by completed local searches. Each
   local search returns its best evaluated point even if it hits its budget.
4. Keep up to `workers` local searches in flight, launching a new one as a
   worker finishes. Newly launched searches use the latest completed local
   results. If a mixed start is infeasible, fall back to its feasible seed.
5. Return the best finite point evaluated anywhere, including screening and
   interrupted local searches. Stop when all seeds have been processed, a
   budget is reached, or the optional `target_value` is attained.

This is an independent implementation of the simplified algorithm, not a port
of the authors' full production code. Extensions here include scrambling,
transformations for infinite bounds, failure handling, transactional persistence,
and asynchronous scheduling. The paper benchmarks Nelder–Mead and DFNLS; our
default is COBYQA, and **we do not implement the paper's DFNLS solver**. See the
[authors' TikTak page](https://www.fatihguvenen.com/tiktak) for their code and the
[SciPy COBYQA documentation](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-cobyqa.html)
for the default local optimizer.

Mixing occurs in transformed coordinates. For finite bounds this is equivalent
to mixing physical parameters; for infinite bounds the map is nonlinear.
`workers=1` with a deterministic objective and fixed seed is reproducible.
Asynchronous completion order can change the search path with multiple workers.
No heuristic claiming repeated local agreement proves global optimality is used.

## Choosing a local optimizer

| `local_method` | Use and limitations |
| --- | --- |
| `"COBYQA"` (default) | Derivative-free quadratic trust-region models; bounds respected. A useful starting point for costly objectives with some local structure. Severe discontinuities or simulation noise can undermine the surrogate. |
| `"pattern"` | Opportunistic coordinate polling; no fitted surrogate or derivatives. Handles invalid trial points and jumps simply, but can stop on plateaus or miss diagonal descent directions. This is not MADS. |
| `"Nelder-Mead"` | Adaptive bounded simplex with an explicit inward initial simplex. Useful comparison to the paper's TikTak-nm; simplex collapse and poor high-dimensional scaling remain possible. |
| `"Powell"` | Derivative-free direction/line searches. Often useful for smoother objectives; costly line searches and holes can hurt performance. |

`local_max_evals` caps new model attempts per local search. Optimizers also have a
callback-count cap of that size (plus the initial feasibility check); cached
calls may make actual model work smaller. `initial_step` and `x_tol` are in
unit-box coordinates. `f_tol` is an absolute improvement threshold for pattern
search and an absolute function tolerance for Nelder–Mead; Powell interprets it
as a relative tolerance. COBYQA stops on trust-region radius, not `f_tol`.
`local_options` forwards additional options to SciPy; it cannot enlarge the
evaluation cap. For discretized models, set tolerances above the numerical noise
floor and compare backends on a limited pilot budget.

Raise `ModelEvaluationError` for expected failures such as equilibrium
nonexistence or iteration nonconvergence. Nonfinite objectives/moments and
`FloatingPointError` are also recorded as failed evaluations and receive an
infinite barrier, so failure can never beat a legitimate large objective.
Unexpected exceptions propagate and are recorded. Add known model exceptions
to `failure_exceptions` explicitly; avoid blanket suppression of programming
errors. A disconnected feasible region can still be missed if screening does
not find it. None of these local algorithms guarantees a solution on an
arbitrary discontinuous objective.

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
Inspect `BoxTransform(...).to_parameters(np.zeros(d))` and its all-ones counterpart
to see the actual computational bounds. Optima outside those cutoffs cannot be
found in that run. If estimates accumulate at a cutoff, revise the transformation
and start a new run with saved estimates as warm starts. Warm starts outside the
cutoff are rejected, not silently clipped.

## Checkpoints, budgets, and reuse

Each run directory contains:

- `history.sqlite3`: configuration, screening points, every distinct evaluated
  unit/physical vector, moment/residual vectors, status/error/runtime, a ledger
  of model-call attempts, and local-search starts/results. SQLite's `-wal` and
  `-shm` companion files may also be present while the run is active.
- `result.json`: a portable best-estimate snapshot, atomically replaced after
  each local search and at return. The database remains authoritative if the
  process dies before the next snapshot.
- `coordinator.lock`: an advisory lock preventing two coordinators from owning
  the same directory. It is released by the operating system when the parent exits.

```python
from dataclasses import replace
from tiktak import load_estimates

# Same model, target, bounds, simulation draws, and algorithm settings:
result = minimize(criterion, bounds, config=replace(config, max_evals=20_000),
                  run_dir="run-v1", problem_id="model-data-draws-v1", resume=True)

# New model/data/settings: reuse coordinates, recompute all objective values.
result = minimize(new_criterion, new_bounds, config=new_config,
                  run_dir="run-v2", problem_id="model-data-draws-v2",
                  warm_start=load_estimates("run-v1", limit=20))
```

Supply the same transformation arguments when resuming. Only `workers`,
`max_evals`, and `max_seconds` may change; other settings are validated. A
versioned `problem_id` is required for explicit run directories. It must change
when model code, targets, solver accuracy, input data, or simulation draws change.
Moment targets/weights/scales are additionally validated automatically. The
package cannot infer arbitrary model-code changes or mutable external inputs.

`max_evals` is a **hard cumulative cap across resumes**, reserved transactionally
before model calls; failed calls count. An interrupted attempt retains its charge
and its point may be retried on resume. Identical parameter points share a cached
evaluation, even across workers; nearby points are never rounded together.
This requires deterministic evaluations, usually with fixed simulation draws
(common random numbers). Failed points are cached too. Change `problem_id` and
use a new run if you need to retry transient failures or change simulation draws.

Completed local searches are skipped on resume. Interrupted local searches replay
from their saved start and use cached evaluations; the internal SciPy trust-region
or simplex state is not serialized. Their prior model attempts still count against
their local allocation. `n_evals` counts attempts, while `n_failed` counts distinct
currently failed/error/abandoned points. `status="completed"` means the allocated
searches finished, not that every local solver converged; inspect `local_results`.
`has_solution` means a finite feasible estimate exists, not global optimality.

`max_seconds` is a **soft per-invocation deadline** checked before new model calls.
Calls already running finish and save their results; an hours-long evaluation can
therefore exceed the deadline by hours. There is no in-process timeout that can
safely kill an arbitrary model. Put timeout-sensitive simulations in their own
managed subprocesses. A parent `callback(result)` runs after local completions for
your progress logging. Do not mutate the active database from callbacks.

## Server execution

Both screening and independent local searches use spawned worker processes.
Parallelism is across model evaluations/local searches, not inside a sequential
local optimizer. Memory use includes one model per worker. Prevent nested BLAS or
model parallelism from oversubscribing the node, for example:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m examples.estimate_ar1 --run-dir /local/persistent-disk/ar1-run \
    --workers 16 --max-evals 3000
python -m examples.estimate_ar1 --run-dir /local/persistent-disk/ar1-run \
    --workers 16 --max-evals 6000 --resume
```

For SLURM, request one node, one task, and the desired `--cpus-per-task`; the
example defaults its worker count from `SLURM_CPUS_PER_TASK`. Allocate enough RAM
for all model copies. Prefer a soft deadline early enough to let running model
calls finish before the scheduler's hard wall time.

**The implemented storage backend supports one host with a local filesystem.**
Do not put the live SQLite database on NFS, Lustre, a syncing folder, or a shared
network drive. An optional `executor=` accepts a caller-owned
`concurrent.futures` executor on the same host (e.g. threads for an external model
that releases the GIL). This is not a multi-node scheduler or a reproduction of
the authors' heterogeneous-machine file-sync implementation.

After a hard job termination, ensure all previous workers have exited before
resuming. The coordinator cannot safely identify orphan workers on your behalf.
To move a run to another node, stop it first and copy the **whole run directory**,
including any SQLite companion files. Ephemeral node scratch disappears after a
job: arrange a backup to persistent storage after workers stop, then restore onto
local disk for the next job. Never treat an actively copied `history.sqlite3`
alone as a consistent backup.
