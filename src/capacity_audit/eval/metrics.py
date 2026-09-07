"""Evaluation metrics: mean, std, CV, CVaR, service level."""
from __future__ import annotations

import numpy as np


def cvar(scenarios: np.ndarray, alpha: float = 0.95) -> float:
    s = np.sort(np.asarray(scenarios, float))
    tail = int(np.ceil(alpha * len(s)))
    return float(np.mean(s[tail - 1:]))


def percentile(scenarios: np.ndarray, q: float) -> float:
    return float(np.percentile(np.asarray(scenarios, float), 100 * q))


def service_level(oos_results: list) -> float:
    """Mean fraction of demand satisfied (1 − unmet/demand)."""
    return 1.0 - float(np.mean([getattr(r, 'mean_unmet_ton', 0) / 100000 for r in oos_results]))
