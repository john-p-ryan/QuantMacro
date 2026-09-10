"""Sobol screening and asynchronous TikTak multistart coordination."""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, field
import json
import math
import multiprocessing
from pathlib import Path
import tempfile
import time
from typing import Callable

import numpy as np
from scipy.stats import qmc

from .local import evaluate_point, run_local
from .objectives import ModelEvaluationError, MomentObjective
from .storage import Store, coordinator_lock
from .transforms import BoxTransform


@dataclass
class TikTakConfig:
    n_samples: int = 256
    n_local: int | None = None
    max_evals: int = 10_000
    local_max_evals: int = 200
    workers: int = 1
    seed: int = 0
    local_method: str = "COBYQA"
    initial_step: float = 0.1
    x_tol: float = 1e-5
    f_tol: float = 1e-8
    mixing_min: float = 0.1
    mixing_max: float = 0.995
    mixing_power: float = 0.5
    max_seconds: float | None = None
    target_value: float | None = None
    local_options: dict = field(default_factory=dict)
    failure_exceptions: tuple = (ModelEvaluationError, FloatingPointError)

    def __post_init__(self):
        for name in ("n_samples", "max_evals", "local_max_evals", "workers"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.n_local is not None and (isinstance(self.n_local, bool)
                                        or not isinstance(self.n_local, int) or self.n_local < 1):
            raise ValueError("n_local must be a positive integer or None")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.local_method not in ("COBYQA", "pattern", "Nelder-Mead", "Powell"):
            raise ValueError("local_method must be COBYQA, pattern, Nelder-Mead, or Powell")
        if not 0 < self.x_tol < self.initial_step <= 0.5:
            raise ValueError("require 0 < x_tol < initial_step <= 0.5 in unit coordinates")
        if not np.isfinite(self.f_tol) or self.f_tol < 0:
            raise ValueError("f_tol must be finite and nonnegative")
        if not 0 <= self.mixing_min <= self.mixing_max < 1:
            raise ValueError("require 0 <= mixing_min <= mixing_max < 1")
        if not np.isfinite(self.mixing_power) or self.mixing_power <= 0:
            raise ValueError("mixing_power must be positive and finite")
        if self.max_seconds is not None and (not np.isfinite(self.max_seconds) or self.max_seconds <= 0):
            raise ValueError("max_seconds must be positive and finite")
        if self.target_value is not None and not np.isfinite(self.target_value):
            raise ValueError("target_value must be finite")
        if (not isinstance(self.failure_exceptions, tuple)
                or not all(isinstance(e, type) and issubclass(e, Exception) for e in self.failure_exceptions)
                or ModelEvaluationError not in self.failure_exceptions):
            raise ValueError("failure_exceptions must be exception classes and include ModelEvaluationError")
        if self.local_method == "pattern" and self.local_options:
            raise ValueError("pattern uses config settings, not local_options")

    def specification(self):
        values = asdict(self)
        # Budgets and worker count can change on resume; algorithm settings cannot.
        for key in ("max_evals", "max_seconds", "workers"):
            values.pop(key)
        values["failure_exceptions"] = [f"{e.__module__}.{e.__qualname__}" for e in self.failure_exceptions]
        return values


@dataclass
class TikTakResult:
    x: np.ndarray | None
    fun: float
    moments: np.ndarray | None
    residuals: np.ndarray | None
    n_evals: int
    n_failed: int
    n_local_completed: int
    status: str
    message: str
    run_dir: str
    local_results: list[dict]

    @property
    def has_solution(self):
        """A feasible estimate was found; this is not a global-optimality certificate."""
        return self.x is not None and np.isfinite(self.fun)

    def to_dict(self):
        result = asdict(self)
        for name in ("x", "moments", "residuals"):
            if result[name] is not None:
                result[name] = result[name].tolist()
        if not np.isfinite(result["fun"]):
            result["fun"] = None
        return result


class _InlineExecutor:
    def submit(self, function, *args):
        future = Future()
        try:
            future.set_result(function(*args))
        except BaseException as exc:
            future.set_exception(exc)
        return future


def _screen(objective, transform, directory, points, config, pool, store):
    values = [None] * len(points)
    pending, next_index, stopped = {}, 0, False
    while pending or (next_index < len(points) and not stopped):
        while not stopped and next_index < len(points) and len(pending) < config.workers:
            i = next_index
            future = pool.submit(evaluate_point, objective, transform, directory,
                                 np.asarray(points[i]), f"screen:{i}", config)
            pending[future] = i
            next_index += 1
        completed, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in completed:
            i = pending.pop(future)
            values[i] = future.result()
            if values[i] is None:
                stopped = True
    if any(value is None for value in values):
        return False
    ranked = sorted((i for i, value in enumerate(values) if np.isfinite(value)), key=lambda i: (values[i], i))
    count = config.n_local if config.n_local is not None else max(1, math.ceil(0.1 * config.n_samples))
    seeds = [points[i] for i in ranked[:count]]
    store.put("seeds", seeds)
    return True


def _target_reached(store, config):
    if config.target_value is None:
        return False
    best = store.best()
    return best is not None and best["value"] <= config.target_value


def _search(objective, transform, directory, seeds, config, pool, store, callback):
    pending = {}
    rows = {row["id"]: row for row in store.local_rows()}
    indices = [i for i in range(len(seeds)) if i not in rows or rows[i]["status"] != "done"]
    position = 0
    while pending or position < len(indices):
        done_results = [json.loads(row["result"]) for row in store.local_rows()
                        if row["status"] == "done" and row["result"] is not None]
        feasible = [result for result in done_results if result["fun"] is not None]
        incumbent = min(feasible, key=lambda result: (result["fun"], result["index"])) if feasible else None
        # Bootstrap the best Sobol seed before opening the parallel pipeline.
        capacity = config.workers if done_results else 1
        stop = store.exhausted() or _target_reached(store, config)
        while not stop and position < len(indices) and len(pending) < capacity:
            i = indices[position]
            position += 1
            if i not in rows:
                seed = np.asarray(seeds[i])
                weight = float(np.clip(((i + 1) / len(seeds)) ** config.mixing_power,
                                       config.mixing_min, config.mixing_max))
                start = seed if incumbent is None else (1 - weight) * seed + weight * np.asarray(incumbent["unit"])
                store.create_local(i, start, seed)
            future = pool.submit(run_local, objective, transform, directory, i, config)
            pending[future] = i
        if not pending:
            break
        completed, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in completed:
            pending.pop(future)
            future.result()  # Unexpected model/programming errors abort visibly.
            snapshot = _result(store, directory, "running", "local search in progress")
            store.export(snapshot.to_dict())
            if callback is not None:
                callback(snapshot)


def _result(store, directory, status, message):
    best = store.best()
    rows = store.local_rows()
    def array(field):
        return None if best is None or best[field] is None else np.asarray(json.loads(best[field]))
    failed = store.connection.execute("SELECT COUNT(*) FROM evaluations WHERE status IN ('failed','error','abandoned')").fetchone()[0]
    return TikTakResult(
        x=array("parameters"), fun=np.inf if best is None else best["value"],
        moments=array("moments"), residuals=array("residuals"), n_evals=store.count(), n_failed=failed,
        n_local_completed=sum(row["status"] == "done" for row in rows), status=status, message=message,
        run_dir=str(directory), local_results=[json.loads(row["result"]) for row in rows if row["result"] is not None])


def minimize(objective: Callable, bounds, *, config: TikTakConfig | None = None,
             scale=None, location=None, tail=1e-6, run_dir=None, problem_id=None,
             resume=False, warm_start=None, executor=None, callback=None) -> TikTakResult:
    """Minimize a scalar objective or ``MomentObjective`` using TikTak.

    ``problem_id`` is your versioned identifier for model code, empirical data,
    simulation draws, and numerical settings. It is required with an explicit
    run directory. Use ``resume=True`` only after all previous workers stopped.
    ``max_evals`` is a cumulative hard model-call cap, including failed and
    interrupted attempts; cached evaluations do not consume it. ``max_seconds``
    is a soft per-invocation deadline checked before launching evaluations.

    The optional executor must implement the concurrent.futures interface and
    run on this same host. It remains owned by the caller. A callback receives
    a result snapshot in the coordinator after each completed local search.
    """
    config = TikTakConfig() if config is None else config
    # Revalidate in case a caller edited a mutable config after construction.
    config.__post_init__()
    transform = BoxTransform(bounds, scale=scale, location=location, tail=tail)
    if not callable(objective):
        raise TypeError("objective must be callable")
    if run_dir is None:
        if resume:
            raise ValueError("resume requires run_dir")
        directory = Path(tempfile.mkdtemp(prefix="tiktak-"))
        problem_id = problem_id or "temporary-run"
    else:
        directory = Path(run_dir).resolve()
        if not isinstance(problem_id, str) or not problem_id.strip():
            raise ValueError("provide a nonempty versioned problem_id with run_dir")
    if resume and warm_start is not None:
        raise ValueError("warm_start is for new runs; resume reuses saved screening points")
    specification = dict(schema=1, problem_id=problem_id, transform=transform.specification(),
                         algorithm=config.specification(),
                         moments=objective.specification() if isinstance(objective, MomentObjective) else None)
    # Validate serializability before starting expensive work.
    specification = json.loads(json.dumps(specification, allow_nan=False))
    deadline = None if config.max_seconds is None else time.time() + config.max_seconds
    with coordinator_lock(directory):
        store = Store(directory)
        pool, owned = executor, False
        try:
            store.initialize(specification, resume=resume, max_evals=config.max_evals, deadline=deadline)
            points = store.get("points")
            if points is None:
                if transform.dimension:
                    engine = qmc.Sobol(transform.dimension, scramble=True, seed=config.seed)
                    points = engine.random_base2((config.n_samples - 1).bit_length())[:config.n_samples].tolist()
                else:
                    points = [[]]
                if warm_start is not None:
                    warm = np.asarray(warm_start, dtype=float)
                    if warm.ndim == 1:
                        warm = warm[None, :]
                    if warm.ndim != 2 or warm.shape[1] != transform.size:
                        raise ValueError("warm_start must contain physical parameter vectors")
                    points.extend(transform.to_unit(x).tolist() for x in warm)
                # Identical fixed/warm points need only one screening task.
                points = [list(point) for point in dict.fromkeys(tuple(point) for point in points)]
                store.put("points", points)
            if pool is None:
                if config.workers == 1:
                    pool = _InlineExecutor()
                else:
                    pool = ProcessPoolExecutor(config.workers, mp_context=multiprocessing.get_context("spawn"))
                    owned = True
            seeds = store.get("seeds")
            screened = seeds is not None or _screen(objective, transform, str(directory), points, config, pool, store)
            seeds = store.get("seeds")
            if screened and seeds and transform.dimension:
                _search(objective, transform, str(directory), seeds, config, pool, store, callback)
            result = _result(store, directory, "completed", "all retained seeds processed; global optimality is not certified")
            if not result.has_solution:
                result.status, result.message = "no_feasible_point", "no finite model evaluation found within the available budget"
            elif _target_reached(store, config):
                result.status, result.message = "target_reached", "requested objective target attained"
            elif not screened or (transform.dimension and result.n_local_completed < len(seeds)):
                result.status, result.message = "budget_exhausted", "evaluation or wall-clock budget exhausted; increase budget and resume"
            store.export(result.to_dict())
            return result
        finally:
            if owned:
                pool.shutdown(wait=True, cancel_futures=True)
            store.close()
