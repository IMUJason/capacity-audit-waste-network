"""Read-only loaders for raw data, each returning tidy DataFrames.

Every loader records the source path; :func:`sha256_file` gives byte-level
provenance. Nothing here mutates the raw files.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import (
    CANDIDATE_SITES,
    CDW_ESTIMATION,
    CITY_CENTERS,
    DISTANCE_CSV,
    DISTANCE_NPY,
    FACILITY_LIST,
    QUZHOU_CLEAN,
    RAW_DATA_SOURCES,
    TRANSPORT_COST,
)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file's bytes (streaming, memory-safe for large CSVs)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return f"sha256:{h.hexdigest()}"


# --- CD&D waste generation time series -------------------------------------
def load_cdw_timeseries(
    province: str | None = None, city: str | None = None
) -> pd.DataFrame:
    """Load city-year C&D waste generation panel (1990-2022).

    Columns: city_code_final, city_name, province, region, year,
    new_building_area_wm2, demolition_area_wm2, cdw_new_ton, cdw_demo_ton,
    cdw_generation_ton, cdw_generation_10kt, longitude, latitude, data_version.
    """
    df = pd.read_csv(CDW_ESTIMATION, encoding="utf-8-sig")
    if province is not None:
        df = df[df["province"] == province]
    if city is not None:
        df = df[df["city_name"] == city]
    return df.reset_index(drop=True)


# --- Candidate facilities ---------------------------------------------------
def load_facility_list(province: str | None = None) -> pd.DataFrame:
    """Load the upstream data-informed candidate-facility parameterization.

    Candidate sites were selected from sampled locations. Capacity and fixed
    cost were generated reproducibly from regional planning parameters and a
    site-specific seeded perturbation; they are not observed facility accounts.

    Columns: site_id, facility_city, facility_lon, facility_lat, province,
    region, avg_direct_distance_km, capacity_min_ton, capacity_max_ton,
    fixed_cost_yuan, capacity_cost_yuan_per_ton, processing_cost_yuan_per_t,
    processing_emission_tco2_per_t, data_version.
    """
    df = pd.read_csv(FACILITY_LIST, encoding="utf-8-sig")
    if province is not None:
        df = df[df["province"] == province]
    return df.reset_index(drop=True)


def load_candidate_sites() -> pd.DataFrame:
    """Load the full candidate-sites sample (1107 sites)."""
    return pd.read_csv(CANDIDATE_SITES, encoding="utf-8-sig")


def load_city_centers() -> pd.DataFrame:
    """Load 371 city-center coordinates."""
    return pd.read_csv(CITY_CENTERS, encoding="utf-8-sig")


# --- Transport & distance ---------------------------------------------------
def load_transport_cost_matrix() -> pd.DataFrame:
    """Load road/rail/water transport cost matrix."""
    return pd.read_csv(TRANSPORT_COST, encoding="utf-8-sig")


def load_distance_matrix_npy() -> np.ndarray:
    """Load the precomputed inter-city distance matrix (.npy)."""
    return np.load(DISTANCE_NPY)


def load_distance_matrix_csv() -> pd.DataFrame:
    """Load the 371-city distance matrix (long form)."""
    return pd.read_csv(DISTANCE_CSV, encoding="utf-8-sig")


# --- Quzhou project-level case ---------------------------------------------
def load_quzhou_clean() -> dict:
    """Load Quzhou real C&D project data (11 projects, 16 candidate sites).

    Units are normalized daily dispatch units (per the file's own metadata);
    moment calibration therefore uses cdw_estimation Zhejiang rows instead.
    """
    import json

    with open(QUZHOU_CLEAN, "r", encoding="utf-8") as f:
        return json.load(f)


# --- Provenance -------------------------------------------------------------
def all_raw_hashes() -> dict:
    """Return {name: {path, sha256, exists, size_bytes}} for every raw source."""
    out: dict[str, dict] = {}
    for name, path in RAW_DATA_SOURCES.items():
        exists = path.exists()
        out[name] = {
            "path": str(path),
            "exists": bool(exists),
            "sha256": sha256_file(path) if exists else None,
            "size_bytes": int(path.stat().st_size) if exists else None,
        }
    return out
