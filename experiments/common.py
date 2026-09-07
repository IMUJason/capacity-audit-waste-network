"""Shared setup for the capacity-audit experiments.

Builds the 40-city / 33-candidate regional facility-location instance
from the frozen data snapshot and exposes the covariance-model builder
and design constants used by every experiment.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np

from tre_dro.ambiguity.covariance import (
    build_spatial_sigma,
    haversine_km_matrix,
)
from tre_dro.ambiguity.spatial_econometrics import build_knn_weight_matrix
from tre_dro.ambiguity.spatiotemporal import (
    fit_pooled_sem,
    fit_trailing_mean_forecast,
    fit_variance_matched_ledoit_wolf,
)
from tre_dro.data.raw_loader import load_cdw_timeseries, load_facility_list
from tre_dro.data.schema import Customer, Facility, Instance, Moments

# --- design constants --------------------------------------------------
DEMAND_PROVINCES = ["江苏省", "浙江省", "安徽省"]
FACILITY_PROVINCES = [
    "上海市", "江苏省", "浙江省", "安徽省", "江西省", "福建省",
    "山东省", "河南省", "湖北省",
]
TRAIN_END = 2012
TEST_START = 2013
TEST_END = 2022
KNN_K = 5
RISK_WEIGHT = 0.5
CVAR_ALPHA = 0.9
DISCOUNT_RATE = 0.05
ASSET_LIFE = 20
CRF = DISCOUNT_RATE * (1 + DISCOUNT_RATE) ** ASSET_LIFE / (
    (1 + DISCOUNT_RATE) ** ASSET_LIFE - 1
)
N_SCENARIOS = 600
SEEDS = (20260714, 20260731, 20260801)
# Layout optimal at the primary setting (theta = 1, baseline demand),
# identical across all four covariance specifications.
CERTIFIED = [0, 1, 2, 3, 4, 9, 10, 11, 12, 16, 21, 22, 23, 24, 30, 31]


def build_instance():
    """Assemble the regional instance from the frozen snapshot."""
    panel = load_cdw_timeseries()
    panel = panel[panel["province"].isin(DEMAND_PROVINCES)].copy()
    cities = sorted(panel["city_name"].unique())
    pivot = panel.pivot(index="year", columns="city_name",
                        values="cdw_generation_ton").reindex(columns=cities)
    if pivot.isna().any().any():
        raise ValueError("Incomplete city-year panel")
    coordinates = (panel.groupby("city_name")[["longitude", "latitude"]]
                   .first().reindex(cities).to_numpy(float))
    values = pivot.to_numpy(float)
    mean, std = values.mean(axis=0), values.std(axis=0, ddof=1)
    customers = tuple(
        Customer(i, coordinates[i, 0], coordinates[i, 1], mean[i], std[i], city)
        for i, city in enumerate(cities)
    )
    frame = load_facility_list()
    frame = frame[frame["province"].isin(FACILITY_PROVINCES)].reset_index(drop=True)
    facilities = tuple(
        Facility(i, float(r.facility_lon), float(r.facility_lat),
                 float(r.fixed_cost_yuan), float(r.capacity_max_ton),
                 f"{r.facility_city}-{r.site_id}",
                 float(r.processing_cost_yuan_per_t))
        for i, r in frame.iterrows()
    )
    fcoords = np.array([[f.lon, f.lat] for f in facilities])
    all_d = haversine_km_matrix(
        np.concatenate([coordinates[:, 0], fcoords[:, 0]]),
        np.concatenate([coordinates[:, 1], fcoords[:, 1]]),
    )
    distances = all_d[:len(customers), len(customers):]
    instance = Instance("yrd_40cities", customers, facilities, distances,
                        1.4, len(facilities), 500.0, {})
    return instance, pivot.index.to_numpy(int), values, coordinates


def fit_forecast(values, years):
    """Trailing-3-year mean forecast on the training window."""
    train = years <= TRAIN_END
    horizon = int(np.sum((years >= TEST_START) & (years <= TEST_END)))
    return fit_trailing_mean_forecast(values[train], horizon, window=3)


def build_covariance_models(trend, coordinates):
    """The four covariance specifications (shared means, differing Sigma)."""
    residual_std = trend.residuals.std(axis=0, ddof=1)
    distances = haversine_km_matrix(coordinates[:, 0], coordinates[:, 1])
    rho = float(np.median(np.sort(distances, axis=1)[:, 1:KNN_K + 1]))
    W = build_knn_weight_matrix(coordinates, k=KNN_K)
    sem = fit_pooled_sem(trend.residuals, W)
    return {
        "Independent": np.diag(residual_std**2),
        "Distance-kernel": build_spatial_sigma(
            Moments(np.ones(trend.residuals.shape[1]), residual_std),
            distances, rho),
        "Shrinkage": fit_variance_matched_ledoit_wolf(trend.residuals),
        "Pooled-SEM": sem.covariance,
    }
