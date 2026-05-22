"""Reference LP radius recomputation sketch for circle packing candidates."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


def recompute_radii_for_centers(centers: np.ndarray) -> np.ndarray:
    """Return radii constrained by square boundaries and pair distances."""
    count = int(centers.shape[0])
    rows: list[np.ndarray] = []
    bounds: list[float] = []
    boundary_caps = np.minimum.reduce(
        [centers[:, 0], centers[:, 1], 1.0 - centers[:, 0], 1.0 - centers[:, 1]]
    )
    for index in range(count):
        row = np.zeros(count)
        row[index] = 1.0
        rows.append(row)
        bounds.append(float(boundary_caps[index]))
    for first in range(count):
        for second in range(first + 1, count):
            row = np.zeros(count)
            row[first] = 1.0
            row[second] = 1.0
            rows.append(row)
            bounds.append(float(np.linalg.norm(centers[first] - centers[second])))
    result = linprog(
        c=-np.ones(count),
        A_ub=np.asarray(rows),
        b_ub=np.asarray(bounds),
        bounds=[(0.0, None)] * count,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(result.message)
    return np.asarray(result.x, dtype=float)
