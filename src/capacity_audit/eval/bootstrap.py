"""Bootstrap confidence intervals for paired method differences."""
from __future__ import annotations

import numpy as np


def bootstrap_ci_diff(
    a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 42,
    confidence: float = 0.95, statistic: str = "mean",
) -> tuple[float, float, dict]:
    """Bootstrap (1-α)·100% CI for the mean difference B − A.

    Returns (ci_lower, ci_upper, diag) where diag includes SE, bias, etc.
    """
    a_f = np.asarray(a, float)
    b_f = np.asarray(b, float)
    rng = np.random.RandomState(seed)
    n = len(a_f)
    boot_diffs = np.zeros(n_boot)
    for j in range(n_boot):
        idx = rng.randint(0, n, size=n)
        if statistic == "mean":
            boot_diffs[j] = float(np.mean(b_f[idx] - a_f[idx]))
        elif statistic == "median":
            boot_diffs[j] = float(np.median(b_f[idx] - a_f[idx]))
    alpha = 1.0 - confidence
    ci_lo = float(np.percentile(boot_diffs, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_diffs, 100 * (1 - alpha / 2)))
    return ci_lo, ci_hi, {
        "n_bootstrap": n_boot,
        "bootstrap_se": float(np.std(boot_diffs, ddof=1)),
        "bootstrap_bias": float(np.mean(boot_diffs)),
        "statistic": statistic,
        "confidence": confidence,
    }
