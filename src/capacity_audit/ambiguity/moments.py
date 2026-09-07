"""Moment estimation: μ_i, σ_i from real city-year panels.

The audit dict returned by each fitter records exactly which data rows produced
each moment — written into ``instances/{case}/calibration_trace.json`` so every
μ_i / σ_i is traceable to source bytes (data-lineage requirement).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..data.schema import Moments


def fit_moments_from_panel(
    panel: pd.DataFrame,
    value_col: str = "cdw_generation_ton",
    group_col: str = "city_name",
    year_window: tuple[int, int] | None = None,
    min_std_frac: float = 0.05,
) -> tuple[Moments, dict[str, Any]]:
    """Per-group mean/std from a city-year panel.

    Returns (Moments, audit) where audit lists the group order, value column,
    year window, and median number of years per group.
    """
    df = panel.copy()
    if "year" in df.columns and year_window is not None:
        df = df[(df["year"] >= year_window[0]) & (df["year"] <= year_window[1])]
    grouped = df.groupby(group_col, sort=True)[value_col]
    means = grouped.mean()
    stds = grouped.std(ddof=1)
    mu = means.to_numpy(dtype=float)
    sigma = stds.to_numpy(dtype=float)
    # Guard degenerate groups (n==1 → NaN std): fall back to a small fraction of μ.
    degenerate = ~np.isfinite(sigma) | (sigma <= 0)
    sigma[degenerate] = mu[degenerate] * min_std_frac
    groups = means.index.tolist()
    audit: dict[str, Any] = {
        "group_col": group_col,
        "value_col": value_col,
        "year_window": year_window,
        "groups_in_order": groups,
        "median_n_years": int(round(df.groupby(group_col)[value_col].count().median())),
        "mu": mu.tolist(),
        "sigma": sigma.tolist(),
        "cv": (sigma / np.maximum(mu, 1e-9)).tolist(),
        "n_degenerate_guarded": int(degenerate.sum()),
    }
    return Moments(mu=mu, sigma=sigma), audit
