"""A restartable server entry point; replace income_moments with your model."""
import argparse
import os
from pathlib import Path

import numpy as np

from tiktak import MomentObjective, TikTakConfig, minimize
from .simple_examples import income_moments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    parser.add_argument("--max-evals", type=int, default=3000)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    target = income_moments([0.65, 0.2])
    objective = MomentObjective(income_moments, target, scales=target)
    config = TikTakConfig(n_samples=64, n_local=8, workers=args.workers,
                          max_evals=args.max_evals, max_seconds=args.max_seconds,
                          local_max_evals=250, x_tol=1e-7)
    result = minimize(objective, [(0, 0.98), (0, np.inf)], scale=[1, 0.2], config=config,
                      run_dir=args.run_dir, problem_id="analytic-ar1-v1", resume=args.resume)
    print(f"{result.status}: loss={result.fun:.8g}, parameters={result.x}, calls={result.n_evals}")
    print(f"Saved to {result.run_dir}")


if __name__ == "__main__":
    main()
