from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
import sqlite3
import time

import numpy as np
import pytest

from tiktak import (BoxTransform, Evaluation, ModelEvaluationError, MomentObjective,
                    TikTakConfig, load_estimates, minimize)
from tiktak.local import Evaluator, evaluate_point, run_local
from tiktak.storage import Store, coordinator_lock


def quadratic(x):
    return float(np.sum((x - np.array([0.2, -0.4])) ** 2))


def rastrigin(x):
    return float(10 * len(x) + np.sum(x * x - 10 * np.cos(2 * np.pi * x)))


def income_moments(x):
    rho, sigma = x
    variance = sigma**2 / (1 - rho**2)
    return np.array([variance, rho * variance, rho**2 * variance])


def rough_holes(x):
    if x[0] < -0.35 or (0.2 < x[0] < 0.45 and x[1] < 0.4):
        raise ModelEvaluationError("no equilibrium")
    if x[1] < -0.7:
        return np.nan
    # Stair steps are a stand-in for discretization of model moments.
    return float(np.sum(np.round((x - np.array([0.6, 0.3])) / 0.02) ** 2) * 0.0004)


def process_objective(x):
    time.sleep(0.02)
    return Evaluation(float(np.sum(x * x)), moments=np.array([os.getpid()]))


def db_rows(path, query):
    with sqlite3.connect(path / "history.sqlite3") as connection:
        return connection.execute(query).fetchall()


@pytest.mark.parametrize("method", ["COBYQA", "pattern", "Nelder-Mead", "Powell"])
def test_local_backends_find_quadratic(tmp_path, method):
    config = TikTakConfig(n_samples=16, n_local=4, local_max_evals=300,
                          max_evals=1500, local_method=method, x_tol=1e-6, f_tol=1e-12)
    result = minimize(quadratic, [(-2, 2)] * 2, config=config,
                      run_dir=tmp_path, problem_id="quadratic-v1")
    assert result.has_solution and result.status == "completed"
    assert result.fun < 1e-8
    np.testing.assert_allclose(result.x, [0.2, -0.4], atol=1e-4)
    assert result.n_evals <= config.max_evals
    saved = json.loads((tmp_path / "result.json").read_text())
    assert saved["fun"] == result.fun


def test_rastrigin_multimodal(tmp_path):
    result = minimize(rastrigin, [(-5.12, 5.12)] * 2,
                      config=TikTakConfig(n_samples=256, n_local=32, local_max_evals=180, seed=7),
                      run_dir=tmp_path, problem_id="rastrigin-v1")
    assert result.fun < 1e-7


def test_moments_with_unbounded_parameter(tmp_path):
    truth = np.array([0.65, 0.2])
    target = income_moments(truth)
    objective = MomentObjective(income_moments, target, scales=target)
    result = minimize(objective, [(0, 0.98), (0, np.inf)], scale=[1, 0.2],
                      config=TikTakConfig(n_samples=64, n_local=8, local_max_evals=250, x_tol=1e-7),
                      run_dir=tmp_path, problem_id="income-v1")
    assert result.fun < 1e-9
    np.testing.assert_allclose(result.x, truth, atol=1e-5)
    np.testing.assert_allclose(result.moments, target, atol=1e-6)
    assert result.fun == pytest.approx(result.residuals @ result.residuals)
    rows = db_rows(tmp_path, "SELECT moments, residuals FROM evaluations WHERE status='ok'")
    assert all(moments is not None and residuals is not None for moments, residuals in rows)


def test_rough_surface_and_expected_failures(tmp_path):
    result = minimize(rough_holes, [(-1, 1)] * 2,
                      config=TikTakConfig(n_samples=128, n_local=16, local_method="pattern",
                                          local_max_evals=150, x_tol=1e-4),
                      run_dir=tmp_path, problem_id="rough-v1")
    assert result.fun < 0.002
    assert result.n_failed > 0
    assert result.has_solution


def test_transform_roundtrip_fixed_finite_and_infinite():
    transform = BoxTransform([(-2, 5), (1, np.inf), (-np.inf, 3), (-np.inf, np.inf), (7, 7)],
                             scale=[1, 2, 4, 3, 1], location=[0, 0, 0, -2, 0])
    for u in [np.zeros(4), np.ones(4), np.array([0.1, 0.3, 0.8, 0.6])]:
        x = transform.to_parameters(u)
        assert np.isfinite(x).all()
        assert x[-1] == 7
        np.testing.assert_allclose(transform.to_unit(x), u, atol=1e-10)
    with pytest.raises(ValueError, match="cutoff"):
        transform.to_unit([0, 1e20, 0, 0, 7])


def test_unbounded_optimum_and_fixed_parameter(tmp_path):
    def objective(x):
        assert x[2] == 3
        return (x[0] + 1.5)**2 + (x[1] - 2)**2
    result = minimize(objective, [(-np.inf, np.inf), (-np.inf, 5), (3, 3)],
                      config=TikTakConfig(n_samples=32, n_local=5, local_max_evals=200),
                      run_dir=tmp_path, problem_id="unbounded-v1")
    assert result.fun < 1e-7


def test_all_fixed_parameters_evaluated_once(tmp_path):
    result = minimize(lambda x: np.sum(x**2), [(2, 2), (3, 3)], run_dir=tmp_path, problem_id="fixed")
    assert result.fun == 13
    assert result.n_evals == 1 and result.n_local_completed == 0
    assert result.status == "completed"


def test_resume_screening_and_cache(tmp_path):
    calls = []
    def objective(x):
        calls.append(x.copy())
        return quadratic(x)
    config = TikTakConfig(n_samples=16, n_local=3, max_evals=7)
    first = minimize(objective, [(-2, 2)] * 2, config=config, run_dir=tmp_path, problem_id="resume-v1")
    assert first.n_evals == 7 and first.status == "budget_exhausted"
    second = minimize(objective, [(-2, 2)] * 2, config=replace(config, max_evals=500),
                      run_dir=tmp_path, problem_id="resume-v1", resume=True)
    assert second.status == "completed" and second.fun < 1e-8
    assert len(calls) == second.n_evals
    before = len(calls)
    third = minimize(objective, [(-2, 2)] * 2, config=replace(config, max_evals=500),
                     run_dir=tmp_path, problem_id="resume-v1", resume=True)
    assert len(calls) == before and third.fun == second.fun


def test_resume_interrupted_local_budget(tmp_path):
    config = TikTakConfig(n_samples=8, n_local=3, max_evals=12, local_max_evals=100)
    first = minimize(quadratic, [(-2, 2)] * 2, config=config, run_dir=tmp_path, problem_id="local-resume")
    assert first.status == "budget_exhausted"
    assert db_rows(tmp_path, "SELECT COUNT(*) FROM locals WHERE status='pending'")[0][0] == 1
    second = minimize(quadratic, [(-2, 2)] * 2, config=replace(config, max_evals=400),
                      run_dir=tmp_path, problem_id="local-resume", resume=True)
    assert second.status == "completed" and second.fun < 1e-8
    assert db_rows(tmp_path, "SELECT MAX(n) FROM (SELECT COUNT(*) n FROM attempts GROUP BY key)")[0][0] == 1


def test_process_parallelism_and_strict_budget(tmp_path):
    result = minimize(process_objective, [(-1, 1)] * 2,
                      config=TikTakConfig(n_samples=64, n_local=4, workers=2, max_evals=13),
                      run_dir=tmp_path, problem_id="process-v1")
    assert result.n_evals == 13
    pids = {json.loads(row[0])[0] for row in db_rows(tmp_path, "SELECT moments FROM evaluations WHERE status='ok'")}
    assert len(pids) == 2 and os.getpid() not in pids


def test_parallel_locals_complete(tmp_path):
    result = minimize(quadratic, [(-2, 2)] * 2,
                      config=TikTakConfig(n_samples=16, n_local=6, workers=2, max_evals=700),
                      run_dir=tmp_path, problem_id="parallel-locals")
    assert result.n_local_completed == 6 and result.fun < 1e-8


def test_exact_cache_deduplicates_concurrent_calls(tmp_path):
    calls = []
    def objective(x):
        calls.append(1)
        time.sleep(0.1)
        return np.sum(x**2)
    store = Store(tmp_path)
    store.initialize({}, resume=False, max_evals=10, deadline=None)
    transform, config = BoxTransform([(-1, 1)]), TikTakConfig()
    def evaluate(i):
        evaluator = Evaluator(objective, transform, tmp_path, f"test:{i}", config)
        try:
            return evaluator(np.array([0.3]))
        finally:
            evaluator.store.close()
    with ThreadPoolExecutor(4) as pool:
        values = list(pool.map(evaluate, range(4)))
    assert values == pytest.approx([0.16] * 4)
    assert len(calls) == store.count() == 1
    store.close()


def test_warm_start_reevaluates_and_rejects_stale_resume(tmp_path):
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    config = TikTakConfig(n_samples=8, n_local=2)
    first = minimize(quadratic, [(-2, 2)] * 2, config=config, run_dir=first_dir, problem_id="v1")
    warm = load_estimates(first_dir, limit=1)
    np.testing.assert_array_equal(warm[0], first.x)
    second = minimize(lambda x: quadratic(x) + 5, [(-2, 2)] * 2, config=config,
                      run_dir=second_dir, problem_id="v2", warm_start=warm)
    assert second.fun >= 5 and second.fun < 5 + 1e-8
    with pytest.raises(ValueError, match="specification"):
        minimize(quadratic, [(-2, 2)] * 2, config=config, run_dir=first_dir, problem_id="v2", resume=True)
    with pytest.raises(FileExistsError):
        minimize(quadratic, [(-2, 2)] * 2, config=config, run_dir=first_dir, problem_id="v1")


def test_weighted_criterion_and_validation():
    target = np.array([1, 2])
    weights = np.array([[2, 0.5], [0.5, 1]])
    objective = MomentObjective(lambda x: x, target, weights=weights, scales=[2, 3])
    x = np.array([2, 4])
    errors = (x - target) / [2, 3]
    assert objective(x).value == pytest.approx(errors @ weights @ errors)
    for bad in [np.array([-1, 1]), np.array([[1, 2], [2, 1]]), np.array([[1, 2], [0, 1]])]:
        with pytest.raises(ValueError):
            MomentObjective(lambda x: x, target, weights=bad)
    with pytest.raises(ValueError):
        MomentObjective(lambda x: [1], target)(x)


def test_unexpected_errors_propagate(tmp_path):
    def broken(x):
        raise TypeError("programming bug")
    with pytest.raises(TypeError, match="programming bug"):
        minimize(broken, [(0, 1)], run_dir=tmp_path, problem_id="broken")
    assert "programming bug" in db_rows(tmp_path, "SELECT error FROM evaluations")[0][0]


def test_no_feasible_solution_and_soft_deadline(tmp_path):
    result = minimize(lambda x: np.inf, [(0, 1)], config=TikTakConfig(n_samples=8),
                      run_dir=tmp_path / "invalid", problem_id="invalid")
    assert not result.has_solution and result.status == "no_feasible_point"
    assert result.n_failed == 8 and result.n_local_completed == 0
    def slow(x):
        time.sleep(0.04)
        return 1.0
    timed = minimize(slow, [(0, 1)], config=TikTakConfig(n_samples=8, max_seconds=0.025),
                     run_dir=tmp_path / "timed", problem_id="timed")
    assert timed.n_evals <= 1
    assert timed.status in ("budget_exhausted", "no_feasible_point")


def test_coordinator_lock_and_invalid_inputs(tmp_path):
    with coordinator_lock(tmp_path):
        with pytest.raises(RuntimeError, match="coordinator"):
            with coordinator_lock(tmp_path):
                pass
    for bounds in [[(2, 1)], [(np.inf, np.inf)], [(np.nan, 1)], []]:
        with pytest.raises(ValueError):
            BoxTransform(bounds)
    with pytest.raises(ValueError):
        TikTakConfig(mixing_max=1)
    with pytest.raises(ValueError):
        minimize(quadratic, [(-2, 2)] * 2, run_dir=tmp_path)


def test_serial_tiktak_mixes_ranked_seeds_with_best_completed_local(tmp_path):
    config = TikTakConfig(n_samples=16, n_local=4, local_max_evals=80)
    result = minimize(rastrigin, [(-5, 5)] * 2, config=config, run_dir=tmp_path, problem_id="mixing")
    with sqlite3.connect(tmp_path / "history.sqlite3") as connection:
        seeds = np.asarray(json.loads(connection.execute("SELECT value FROM meta WHERE key='seeds'").fetchone()[0]))
        starts = [np.asarray(json.loads(row[0])) for row in connection.execute("SELECT start FROM locals ORDER BY id")]
    np.testing.assert_array_equal(starts[0], seeds[0])
    for i in range(1, len(seeds)):
        best = min(result.local_results[:i], key=lambda r: (r["fun"], r["index"]))
        weight = min(0.995, max(0.1, np.sqrt((i + 1) / len(seeds))))
        np.testing.assert_allclose(starts[i], (1 - weight) * seeds[i] + weight * np.asarray(best["unit"]))


def test_infeasible_blend_falls_back_to_seed(tmp_path):
    calls = []
    def objective(x):
        calls.append(x[0])
        if 0.3 < x[0] < 0.7:
            raise ModelEvaluationError("hole at mixed start")
        return (x[0] - 0.9)**2
    store = Store(tmp_path)
    store.initialize({}, resume=False, max_evals=100, deadline=None)
    transform = BoxTransform([(0, 1)])
    config = TikTakConfig(local_method="pattern", local_max_evals=50)
    evaluate_point(objective, transform, tmp_path, np.array([0.9]), "seed", config)
    store.create_local(0, np.array([0.5]), np.array([0.9]))
    result = run_local(objective, transform, tmp_path, 0, config)
    assert calls[:2] == pytest.approx([0.9, 0.5])
    assert result["fun"] == 0 and result["x"] == [0.9]
    assert sum(x == 0.9 for x in calls) == 1
    store.close()


def test_interrupt_is_retryable_and_retains_budget_charge(tmp_path):
    interrupted = False
    def objective(x):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt()
        return float(x @ x)
    config = TikTakConfig(n_samples=4, n_local=1, local_max_evals=30)
    with pytest.raises(KeyboardInterrupt):
        minimize(objective, [(-1, 1)], config=config, run_dir=tmp_path, problem_id="interrupt")
    assert db_rows(tmp_path, "SELECT status FROM evaluations")[0][0] == "abandoned"
    result = minimize(objective, [(-1, 1)], config=config, run_dir=tmp_path, problem_id="interrupt", resume=True)
    unique = db_rows(tmp_path, "SELECT COUNT(*) FROM evaluations")[0][0]
    assert result.has_solution and result.n_evals == unique + 1


@pytest.mark.parametrize("method", ["COBYQA", "pattern", "Nelder-Mead", "Powell"])
def test_boundary_optimum_and_warm_seed(tmp_path, method):
    def objective(x):
        assert np.all(x >= 0) and np.all(x <= 1)
        return (x[0] + 1)**2 + (x[1] - 2)**2
    result = minimize(objective, [(0, 1)] * 2, warm_start=[[0, 1]],
                      config=TikTakConfig(n_samples=8, n_local=2, local_method=method),
                      run_dir=tmp_path, problem_id="boundary")
    np.testing.assert_array_equal(result.x, [0, 1])
    assert result.fun == 2
