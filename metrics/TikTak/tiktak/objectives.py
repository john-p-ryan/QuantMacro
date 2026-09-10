"""Model-independent method-of-moments criterion."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


class ModelEvaluationError(Exception):
    """Expected economic/numerical infeasibility, e.g. no equilibrium exists."""


@dataclass
class Evaluation:
    value: float
    moments: np.ndarray | None = None
    residuals: np.ndarray | None = None


class MomentObjective:
    """Wrap ``model(theta) -> moments`` as (m(theta)-target)' W (m(theta)-target).

    ``scales`` standardizes moment errors before applying W. W can be a
    nonnegative diagonal vector or a symmetric positive-semidefinite matrix.
    Identity weights and unit scales are defaults. No covariance matrix is
    silently estimated or inverted. Use fixed simulation draws for caching.
    """

    def __init__(self, model: Callable, target, *, weights=None, scales=None):
        self.model = model
        self.target = np.asarray(target, dtype=float).copy()
        if self.target.ndim != 1 or self.target.size == 0 or not np.isfinite(self.target).all():
            raise ValueError("target must be a nonempty finite vector")
        n = self.target.size
        self.scales = np.broadcast_to(1.0 if scales is None else scales, (n,)).copy()
        if not np.isfinite(self.scales).all() or np.any(self.scales <= 0):
            raise ValueError("moment scales must be positive and finite")
        w = np.ones(n) if weights is None else np.asarray(weights, dtype=float).copy()
        if not np.isfinite(w).all():
            raise ValueError("weights must be finite")
        if w.shape == (n,):
            if np.any(w < 0):
                raise ValueError("diagonal weights cannot be negative")
            self.factor = np.sqrt(w)
        elif w.shape == (n, n):
            if not np.allclose(w, w.T, rtol=1e-10, atol=1e-12):
                raise ValueError("weight matrix must be symmetric")
            eigenvalues, vectors = np.linalg.eigh((w + w.T) / 2)
            if eigenvalues.min() < -1e-12 * max(1.0, np.max(np.abs(eigenvalues))):
                raise ValueError("weight matrix must be positive semidefinite")
            self.factor = np.sqrt(np.maximum(eigenvalues, 0))[:, None] * vectors.T
        else:
            raise ValueError("weights must be a diagonal vector or square matrix")
        self.weights = w

    def __call__(self, parameters):
        moments = np.asarray(self.model(parameters), dtype=float)
        if moments.shape != self.target.shape:
            raise ValueError("model moments have a different shape than target")
        if not np.isfinite(moments).all():
            raise ModelEvaluationError("model returned nonfinite moments")
        errors = (moments - self.target) / self.scales
        residuals = self.factor * errors if self.factor.ndim == 1 else self.factor @ errors
        return Evaluation(float(residuals @ residuals), moments, residuals)

    def specification(self):
        return dict(target=self.target.tolist(), weights=self.weights.tolist(),
                    scales=self.scales.tolist())
