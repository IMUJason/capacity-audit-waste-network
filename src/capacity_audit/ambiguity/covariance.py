"""Spatial covariance matrix for the joint moment ambiguity set.

    Sigma_ij = sigma_i * sigma_j * exp(-h_ij / rho)   (exponential kernel)

Properties:
- PSD by construction: exp(-h/rho) is positive-definite whenever h is a
  conditionally negative-definite distance (haversine great-circle
  distance is).
- Captures spatial demand correlation: nearby locations co-move, so a
  demand shock hits correlated clusters.
- For rho -> 0 or diagonal use, Sigma degenerates to diag(sigma^2)
  (independent).
"""
from __future__ import annotations

import numpy as np

from ..data.schema import Moments


def haversine_km_matrix(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Pairwise great-circle distances (km). ``lon``, ``lat`` are (n,) arrays."""
    lon = np.radians(np.asarray(lon, float))
    lat = np.radians(np.asarray(lat, float))
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    R = 6371.0088  # mean Earth radius, km
    return 2.0 * R * np.arcsin(np.sqrt(a))


def build_spatial_sigma(
    moments: Moments,
    haversine_km: np.ndarray,
    rho_km: float,
    kernel: str = "exponential",
    floor_eig: float = 1e-10,
) -> np.ndarray:
    """Σ_ij = σ_i σ_j · K(h_ij); PSD-floored; returns (n,n)."""
    sigma = np.asarray(moments.sigma, float)
    h = np.asarray(haversine_km, float)
    rho = max(float(rho_km), 1e-9)
    if kernel == "exponential":
        K = np.exp(-h / rho)
    elif kernel == "matern32":
        r = h / rho
        K = (1.0 + np.sqrt(3.0) * r) * np.exp(-np.sqrt(3.0) * r)
    else:
        raise ValueError(f"unknown kernel: {kernel}")
    np.fill_diagonal(K, 1.0)
    Sigma = (sigma[:, None] * K) * sigma[None, :]
    return _psd_floor(Sigma, floor_eig)


def diagonal_sigma(moments: Moments) -> np.ndarray:
    """Σ = diag(σ²) — the *independent* (no spatial correlation) variant."""
    s = np.asarray(moments.sigma, float)
    return np.diag(s ** 2)


def homogeneous_sigma(moments: Moments) -> np.ndarray:
    """Σ using a single global CV — the *homogeneous-DRO* variant (Σ = diag of σ̄²)."""
    cv_global = float(np.mean(moments.cv))
    mu = np.asarray(moments.mu, float)
    sigma_global = mu * cv_global
    return np.diag(sigma_global ** 2)


def cholesky_sqrt(sigma_matrix: np.ndarray) -> np.ndarray:
    """Σ^{1/2} = V diag(√λ) Vᵀ — the symmetric PSD square root.

    Uses eigendecomposition (works for any PSD matrix, not just those with a
    Cholesky factor).  The symmetric square root guarantees
    (Σ^{1/2})(Σ^{1/2}) = Σ and ‖Σ^{1/2} π‖₂ = √(πᵀ Σ π).
    """
    M = 0.5 * (sigma_matrix + sigma_matrix.T)
    w, V = np.linalg.eigh(M)
    w = np.clip(w, 0.0, None)
    # V * √w  broadcasts column-wise = V @ diag(√w)
    L = (V * np.sqrt(w)) @ V.T
    return 0.5 * (L + L.T)  # guard symmetry


def robust_norm(sigma_half: np.ndarray, pi: np.ndarray) -> float:
    """‖Σ^{1/2} π‖₂ — the core worst-case term.

    Re-activates and generalises ``robust_cuts.compute_robust_norm`` (which was
    written correctly but disabled at ``benders_dro.py:548``) from diagonal Σ
    to a full spatial covariance.
    """
    v = sigma_half @ np.asarray(pi, float)
    return float(np.sqrt(v @ v))


def assert_psd(sigma_matrix: np.ndarray, tol: float = -1e-10) -> None:
    """Raise if Σ has a negative eigenvalue below ``tol``."""
    w = np.linalg.eigvalsh(0.5 * (sigma_matrix + sigma_matrix.T))
    assert w.min() >= tol, f"Σ not PSD: min eigenvalue {w.min():.3e} < tol {tol}"


def min_eigval(sigma_matrix: np.ndarray) -> float:
    w = np.linalg.eigvalsh(0.5 * (sigma_matrix + sigma_matrix.T))
    return float(w.min())


def condition_number(sigma_matrix: np.ndarray) -> float:
    w = np.linalg.eigvalsh(0.5 * (sigma_matrix + sigma_matrix.T))
    w = np.clip(w, 1e-300, None)
    return float(w.max() / w.min())


def rescale_to_sample_variances(
    sigma_matrix: np.ndarray, sample_variances: np.ndarray
) -> np.ndarray:
    """Rescale a covariance matrix to match given sample variances on the diagonal.

    Extracts the correlation matrix R_ij = Σ_ij / √(Σ_ii·Σ_jj),
    then rescales: Σ'_ij = σ̂_i · σ̂_j · R_ij.

    This preserves:
    - The correlation structure (R is unchanged)
    - PSD property (diagonal rescaling of a PSD matrix preserves PSD)
    - The diagonal now exactly matches the given sample variances.
    """
    M = 0.5 * (np.asarray(sigma_matrix, float) + np.asarray(sigma_matrix, float).T)
    sigma_diag = np.sqrt(np.maximum(np.diag(M), 1e-30))
    R = M / (sigma_diag[:, None] * sigma_diag[None, :])
    R = _psd_floor(R, floor_eig=1e-12)
    # Re-normalize after the numerical PSD projection. Congruence by a positive
    # diagonal matrix preserves PSD and restores a unit correlation diagonal.
    r_diag = np.sqrt(np.maximum(np.diag(R), 1e-30))
    R = R / (r_diag[:, None] * r_diag[None, :])
    np.fill_diagonal(R, 1.0)
    target_variances = np.asarray(sample_variances, float)
    if target_variances.shape != (M.shape[0],) or np.any(target_variances < 0):
        raise ValueError("sample_variances must be a non-negative vector matching Sigma")
    target_std = np.sqrt(target_variances)
    out = (target_std[:, None] * R) * target_std[None, :]
    return 0.5 * (out + out.T)


def _psd_floor(sigma_matrix: np.ndarray, floor_eig: float) -> np.ndarray:
    """Project to nearest PSD matrix with eigenvalues ≥ floor_eig."""
    M = 0.5 * (sigma_matrix + sigma_matrix.T)
    w, V = np.linalg.eigh(M)
    w_floor = np.maximum(w, floor_eig)
    out = (V * w_floor) @ V.T
    return 0.5 * (out + out.T)
