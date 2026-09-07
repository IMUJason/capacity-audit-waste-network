"""Numerical diagnostics: PSD checks, CPLEX IIS recovery, dual validation."""
from __future__ import annotations

import numpy as np


def assert_psd(matrix: np.ndarray, tol: float = -1e-10, name: str = "matrix") -> None:
    M = 0.5 * (matrix + matrix.T)
    w = np.linalg.eigvalsh(M)
    if w.min() < tol:
        raise ValueError(f"{name} not PSD: min eig {w.min():.2e} < {tol}")


def is_psd(matrix: np.ndarray, tol: float = -1e-10) -> bool:
    M = 0.5 * (matrix + matrix.T)
    return bool(np.linalg.eigvalsh(M).min() >= tol)


def solve_time_minutes(result: object) -> float:
    """Extract solve_time in minutes, handling None."""
    wall = getattr(result, "solve_time", None) or 0.0
    return float(wall) / 60.0
