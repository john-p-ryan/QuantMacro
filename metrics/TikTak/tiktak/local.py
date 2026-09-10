"""Failure-aware local search workers; no model-specific logic lives here."""
from __future__ import annotations

import json
import time

import numpy as np
from scipy.optimize import minimize

from .objectives import Evaluation, ModelEvaluationError
from .storage import BudgetExhausted, LocalBudgetExhausted, Store


class Evaluator:
    def __init__(self, objective, transform, directory, task, config, local=False):
        self.objective = objective
        self.transform = transform
        self.store = Store(directory)
        self.task = task
        self.limit = config.local_max_evals if local else None
        self.failure_exceptions = config.failure_exceptions

    def __call__(self, unit):
        # Supported optimizers respect bounds; tolerate only floating-point drift.
        unit = np.asarray(unit, dtype=float)
        if np.any(unit < -1e-12) or np.any(unit > 1 + 1e-12):
            raise ValueError("local optimizer proposed a point outside the unit box")
        unit = np.clip(unit, 0, 1)
        parameters = self.transform.to_parameters(unit)
        key, cached = self.store.claim(unit, parameters, self.task, self.limit)
        if cached is not None:
            return float(cached["value"]) if cached["status"] == "ok" else np.inf
        started = time.monotonic()
        try:
            output = self.objective(parameters.copy())
            evaluation = output if isinstance(output, Evaluation) else Evaluation(float(output))
            value = float(evaluation.value)
            if not np.isfinite(value):
                raise ModelEvaluationError("objective returned a nonfinite value")
            for field in ("moments", "residuals"):
                array = getattr(evaluation, field)
                if array is not None:
                    array = np.asarray(array, dtype=float)
                    if array.ndim != 1:
                        raise ValueError(f"Evaluation.{field} must be a vector")
                    if not np.isfinite(array).all():
                        raise ModelEvaluationError(f"nonfinite {field}")
                    setattr(evaluation, field, array)
        except self.failure_exceptions as exc:
            self.store.finish(key, error=f"{type(exc).__name__}: {exc}", seconds=time.monotonic() - started)
            return np.inf
        except Exception as exc:
            self.store.finish(key, error=f"{type(exc).__name__}: {exc}",
                              seconds=time.monotonic() - started, unexpected=True)
            raise
        except BaseException:
            # Interruptions are retryable on resume, but retain the budget charge.
            self.store.connection.execute("UPDATE evaluations SET status='abandoned' WHERE key=?", (key,))
            raise
        self.store.finish(key, value=value, moments=evaluation.moments, residuals=evaluation.residuals,
                          seconds=time.monotonic() - started)
        return value


def evaluate_point(objective, transform, directory, unit, task, config):
    evaluator = Evaluator(objective, transform, directory, task, config)
    try:
        try:
            return evaluator(unit)
        except BudgetExhausted:
            return None
    finally:
        evaluator.store.close()


def pattern_search(fun, start, *, step, tolerance, improvement_tol, max_calls):
    """Opportunistic coordinate polling with mesh contraction.

    No derivatives or interpolating models. This conservative fallback can
    handle jumps/invalid regions but is not a MADS implementation and offers
    no global guarantee on discontinuous or disconnected feasible sets.
    """
    x = start.copy()
    best = fun(x)
    calls = 1
    while step > tolerance and calls < max_calls:
        improved = False
        for j in range(len(x)):
            for direction in (1, -1):
                trial = x.copy()
                trial[j] = np.clip(x[j] + direction * step, 0, 1)
                if np.array_equal(trial, x):
                    continue
                value = fun(trial)
                calls += 1
                if value < best - improvement_tol:
                    x, best, improved = trial, value, True
                    break
                if calls >= max_calls:
                    break
            if improved or calls >= max_calls:
                break
        if not improved:
            step *= 0.5
    return step <= tolerance, "mesh tolerance reached" if step <= tolerance else "local call limit reached"


def run_local(objective, transform, directory, index, config):
    task = f"local:{index}"
    evaluator = Evaluator(objective, transform, directory, task, config, local=True)
    store = evaluator.store
    row = store.connection.execute("SELECT * FROM locals WHERE id=?", (index,)).fetchone()
    start, seed = np.asarray(json.loads(row["start"])), np.asarray(json.loads(row["seed"]))
    converged, complete, message = False, True, ""
    try:
        try:
            # The mixing segment can pass through an economic infeasibility hole.
            # The unblended Sobol seed is known feasible and is a safe fallback.
            if not np.isfinite(evaluator(start)):
                start = seed
            if config.local_method == "pattern":
                converged, message = pattern_search(
                    evaluator, start, step=config.initial_step, tolerance=config.x_tol,
                    improvement_tol=config.f_tol, max_calls=config.local_max_evals)
            else:
                options = dict(maxfev=config.local_max_evals)
                if config.local_method == "COBYQA":
                    options.update(initial_tr_radius=config.initial_step, final_tr_radius=config.x_tol)
                elif config.local_method == "Nelder-Mead":
                    # SciPy's relative default simplex is tiny near zero and can
                    # collapse at a bound. Build a full-rank inward simplex.
                    simplex = np.tile(start, (len(start) + 1, 1))
                    for j in range(len(start)):
                        sign = 1 if start[j] <= 0.5 else -1
                        simplex[j + 1, j] += sign * min(config.initial_step, 0.5)
                    options.update(xatol=config.x_tol, fatol=config.f_tol,
                                   adaptive=True, initial_simplex=simplex)
                elif config.local_method == "Powell":
                    options.update(xtol=config.x_tol, ftol=config.f_tol)
                options.update(config.local_options)
                # Enforce the caller's cap even if local_options sets maxfev.
                options["maxfev"] = config.local_max_evals
                result = minimize(evaluator, start, method=config.local_method,
                                  bounds=[(0.0, 1.0)] * len(start), options=options)
                converged, message = bool(result.success), str(result.message)
        except LocalBudgetExhausted:
            message = "local evaluation budget exhausted"
        except BudgetExhausted:
            complete, message = False, "global evaluation or wall-clock budget exhausted"
        best = store.best(task)
        result = dict(index=index, unit=None if best is None else json.loads(best["unit"]),
                      x=None if best is None else json.loads(best["parameters"]),
                      fun=None if best is None else best["value"],
                      converged=converged and best is not None, message=message)
        store.finish_local(index, result, complete)
        return result
    finally:
        store.close()
