"""Coordinate maps between economic parameters and a finite optimization box."""
from __future__ import annotations

import numpy as np


class BoxTransform:
    """Map free parameters to [0, 1]; remove fixed parameters internally.

    Finite bounds use an affine map. One-sided bounds use a rational map;
    two-sided infinite bounds use a tangent map. ``tail`` truncates only the
    infinite ends to keep actual model inputs finite. ``scale`` sets the unit
    of exploration on unbounded dimensions; ``location`` centers fully
    unbounded dimensions. Both are in physical parameter units.
    """

    def __init__(self, bounds, *, scale=None, location=None, tail=1e-6):
        b = np.asarray(bounds, dtype=float)
        if b.ndim != 2 or b.shape[1] != 2 or len(b) == 0:
            raise ValueError("bounds must have shape (number of parameters, 2)")
        self.lower, self.upper = b[:, 0].copy(), b[:, 1].copy()
        if (np.isnan(b).any() or np.any(self.lower > self.upper)
                or np.isposinf(self.lower).any() or np.isneginf(self.upper).any()):
            raise ValueError("invalid parameter bounds")
        if not np.isfinite(tail) or not 0 < tail < 0.5:
            raise ValueError("tail must lie strictly between 0 and 0.5")
        self.tail = float(tail)
        self.scale = np.broadcast_to(1.0 if scale is None else scale, (len(b),)).copy()
        self.location = np.broadcast_to(0.0 if location is None else location, (len(b),)).copy()
        if not np.isfinite(self.scale).all() or np.any(self.scale <= 0):
            raise ValueError("scale must be finite and positive")
        if not np.isfinite(self.location).all():
            raise ValueError("location must be finite")
        self.free = self.lower != self.upper
        self.dimension = int(self.free.sum())
        self.size = len(b)
        # Fail before launching a model if user-specified scales overflow.
        if not (np.isfinite(self.to_parameters(np.zeros(self.dimension))).all()
                and np.isfinite(self.to_parameters(np.ones(self.dimension))).all()):
            raise ValueError("transformed bounds overflow; reduce scale or increase tail")

    def to_parameters(self, unit):
        u = np.asarray(unit, dtype=float)
        if u.shape != (self.dimension,) or not np.isfinite(u).all():
            raise ValueError("unit point has invalid shape or nonfinite coordinates")
        if np.any(u < 0) or np.any(u > 1):
            raise ValueError("unit point lies outside [0, 1]")
        x = self.lower.copy()
        for j, v in zip(np.flatnonzero(self.free), u):
            lo, hi, s = self.lower[j], self.upper[j], self.scale[j]
            if np.isfinite(lo) and np.isfinite(hi):
                x[j] = (1 - v) * lo + v * hi
            elif np.isfinite(lo):
                z = (1 - self.tail) * v
                x[j] = lo + s * z / (1 - z)
            elif np.isfinite(hi):
                z = (1 - self.tail) * (1 - v)
                x[j] = hi - s * z / (1 - z)
            else:
                z = self.tail + (1 - 2 * self.tail) * v
                x[j] = self.location[j] + s * np.tan(np.pi * (z - 0.5))
        return x

    def to_unit(self, parameters):
        x = np.asarray(parameters, dtype=float)
        if x.shape != (self.size,) or not np.isfinite(x).all():
            raise ValueError("parameters must be a finite vector matching bounds")
        if np.any(x < self.lower) or np.any(x > self.upper):
            raise ValueError("parameters violate bounds")
        values = []
        for j in np.flatnonzero(self.free):
            lo, hi, s = self.lower[j], self.upper[j], self.scale[j]
            if np.isfinite(lo) and np.isfinite(hi):
                with np.errstate(over="ignore"):
                    width = hi - lo
                # Avoid overflow for very wide intervals, but preserve tiny ones.
                v = ((x[j] - lo) / width if np.isfinite(width)
                     else (x[j] / 2 - lo / 2) / (hi / 2 - lo / 2))
            elif np.isfinite(lo):
                d = (x[j] - lo) / s
                v = (1 - 1 / (1 + d)) / (1 - self.tail)
            elif np.isfinite(hi):
                d = (hi - x[j]) / s
                v = 1 - (1 - 1 / (1 + d)) / (1 - self.tail)
            else:
                z = 0.5 + np.arctan((x[j] - self.location[j]) / s) / np.pi
                v = (z - self.tail) / (1 - 2 * self.tail)
            values.append(v)
        u = np.asarray(values)
        if np.any(u < -1e-12) or np.any(u > 1 + 1e-12):
            raise ValueError("parameters exceed the finite tail cutoff")
        return np.clip(u, 0, 1)

    def specification(self):
        bounds = [[float(v) if np.isfinite(v) else str(v) for v in pair]
                  for pair in zip(self.lower, self.upper)]
        return dict(bounds=bounds,
                    scale=self.scale.tolist(), location=self.location.tolist(), tail=self.tail)
