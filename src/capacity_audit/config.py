"""Configuration: paths and constants for the regional facility-location
audit package.

Raw inputs are frozen snapshots in ``data/raw/`` (read-only). The
package never mutates raw data; ``data.raw_loader`` provides read-only
access with SHA-256 provenance so every result traces to exact bytes.
"""
from __future__ import annotations

from pathlib import Path

# --- Anchored paths ---------------------------------------------------------
# .../<repo>/src/capacity_audit/config.py
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]

DATA_RAW_DIR = REPO_ROOT / "data" / "raw"

# --- Raw data sources (frozen snapshot in data/raw/, READ-ONLY) -------------
CDW_ESTIMATION = DATA_RAW_DIR / "cdw_estimation_1990_2022.csv"
FACILITY_LIST = DATA_RAW_DIR / "facility_list.csv"
TRANSPORT_COST = DATA_RAW_DIR / "transport_cost_matrix.csv"
DISTANCE_NPY = DATA_RAW_DIR / "distance_matrix.npy"
DISTANCE_CSV = DATA_RAW_DIR / "distance_matrix_calculated.csv"
CANDIDATE_SITES = DATA_RAW_DIR / "candidate_sites_sampled.csv"
CITY_CENTERS = DATA_RAW_DIR / "city_centers.csv"
QUZHOU_CLEAN = DATA_RAW_DIR / "quzhou_clean.json"

# Registry of raw inputs used for SHA-256 provenance bookkeeping.
RAW_DATA_SOURCES: dict[str, Path] = {
    "cdw_estimation_1990_2022": CDW_ESTIMATION,
    "facility_list": FACILITY_LIST,
}

# --- Output directories -----------------------------------------------------
RESULTS_DIR = REPO_ROOT / "results"

PACKAGE_VERSION = "1.0.0"

# --- Case study: Zhejiang province (11 prefecture-level cities) ------------
ZHEJIANG_CITIES = [
    "杭州市", "宁波市", "温州市", "嘉兴市", "湖州市",
    "绍兴市", "金华市", "衢州市", "舟山市", "台州市", "丽水市",
]

# CV observed in the real Zhejiang/Quzhou generation series (~0.85); used to
# anchor synthetic experiments at a realistic heterogeneity level.
REALISTIC_CV_ANCHOR = 0.85
