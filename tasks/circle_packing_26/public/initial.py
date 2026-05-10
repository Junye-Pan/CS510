"""Seed candidate for the circle_packing_26 task."""

import numpy as np


def construct_packing():
    """Construct a simple valid seed layout."""
    n = 26
    centers = np.zeros((n, 2))
    centers[0] = [0.5, 0.5]

    for index in range(8):
        angle = 2 * np.pi * index / 8
        centers[index + 1] = [0.5 + 0.3 * np.cos(angle), 0.5 + 0.3 * np.sin(angle)]

    for index in range(16):
        angle = 2 * np.pi * index / 16
        centers[index + 9] = [0.5 + 0.7 * np.cos(angle), 0.5 + 0.7 * np.sin(angle)]

    centers = np.clip(centers, 0.01, 0.99)
    radii = compute_max_radii(centers)
    return centers, radii


def compute_max_radii(centers):
    """Compute a simple feasible radius assignment."""
    n = centers.shape[0]
    radii = np.ones(n)

    for index in range(n):
        x_coord, y_coord = centers[index]
        radii[index] = min(x_coord, y_coord, 1 - x_coord, 1 - y_coord)

    for first in range(n):
        for second in range(first + 1, n):
            dist = np.sqrt(np.sum((centers[first] - centers[second]) ** 2))
            if radii[first] + radii[second] > dist:
                scale = dist / (radii[first] + radii[second])
                radii[first] *= scale
                radii[second] *= scale

    return radii


def run_packing():
    """Return a valid seed packing candidate."""
    centers, radii = construct_packing()
    return centers, radii, float(np.sum(radii))
