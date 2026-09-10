"""Run from the TikTak folder: python -m examples.simple_examples --workers 2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tiktak import ModelEvaluationError, MomentObjective, TikTakConfig, minimize


def quadratic(x):
    return float(np.sum((x - np.array([0.2, -0.4]))**2))


def rastrigin(x):
    return float(10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


def income_moments(x):
    """Stationary AR(1) variance and first two autocovariances."""
    rho, sigma = x
    if abs(rho) >= 1 or sigma < 0:
        raise ModelEvaluationError("no stationary income distribution")
    variance = sigma**2 / (1 - rho**2)
    return np.array([variance, rho * variance, rho**2 * variance])


def rough_holes(x):
    if x[0] < -0.35 or (0.2 < x[0] < 0.45 and x[1] < 0.4):
        raise ModelEvaluationError("no equilibrium")
    if x[1] < -0.7:
        return np.nan
    return float(np.sum(np.round((x - np.array([0.6, 0.3])) / 0.02)**2) * 0.0004)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("tiktak-runs/simple"))
    args = parser.parse_args()
    target = income_moments([0.65, 0.2])
    cases = [
        ("quadratic", quadratic, [(-2, 2)] * 2, dict(n_samples=16, n_local=4), {}),
        ("rastrigin", rastrigin, [(-5.12, 5.12)] * 2,
         dict(n_samples=256, n_local=32, local_max_evals=180, seed=7), {}),
        ("ar1_moments", MomentObjective(income_moments, target, scales=target),
         [(0, 0.98), (0, np.inf)], dict(n_samples=64, n_local=8, local_max_evals=250, x_tol=1e-7),
         dict(scale=[1, 0.2])),
        ("rough_holes", rough_holes, [(-1, 1)] * 2,
         dict(n_samples=128, n_local=16, local_method="pattern", local_max_evals=150, x_tol=1e-4), {}),
    ]
    summary = {}
    for name, objective, bounds, settings, options in cases:
        config = TikTakConfig(workers=args.workers, **settings)
        result = minimize(objective, bounds, config=config, run_dir=args.output / name,
                          problem_id=f"example-{name}-v1", **options)
        summary[name] = result.to_dict()
        print(f"{name:14s} loss={result.fun:.6g}  x={result.x}  "
              f"evaluations={result.n_evals}  failures={result.n_failed}  {result.status}")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    # Required for multiprocessing's spawn mode on macOS/Linux.
    main()
