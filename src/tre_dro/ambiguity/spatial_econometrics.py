"""Spatial econometric diagnostics and model estimation for DRO calibration.

Implements Moran's I, spatial weight matrices, Spatial Error Model (SEM),
and hierarchical covariance decomposition — all from scratch with numpy/scipy
(no pysal/spreg/spdep dependency).

Key references:
- Anselin L. (1988) Spatial Econometrics. Springer.
- LeSage J, Pace RK. (2009) Introduction to Spatial Econometrics. CRC.
- Elhorst JP. (2014) Spatial Econometrics: From Cross-Sectional to Spatial Panels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

from ..data.schema import Moments


# ============================================================================
# Spatial Weight Matrices
# ============================================================================

def build_knn_weight_matrix(
    coords: np.ndarray,  # (n, 2) [lon, lat]
    k: int = 5,
    row_standardize: bool = True,
) -> np.ndarray:
    """K-nearest-neighbour spatial weight matrix W (n,n)."""
    n = coords.shape[0]
    k = min(k, n - 1)
    dist = cdist(coords, coords, metric='euclidean')
    W = np.zeros((n, n))
    for i in range(n):
        nn = np.argsort(dist[i])[1:k + 1]  # exclude self
        W[i, nn] = 1.0
    if row_standardize:
        row_sum = W.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum == 0, 1.0, row_sum)
        W /= row_sum
    return W


def build_distance_threshold_weight_matrix(
    coords: np.ndarray,
    threshold_km: float,
    row_standardize: bool = True,
) -> np.ndarray:
    """Binary spatial weight matrix: W_ij = 1 if dist < threshold."""
    dist = cdist(coords, coords, metric='euclidean')
    W = (dist < threshold_km).astype(float)
    np.fill_diagonal(W, 0.0)
    if row_standardize:
        row_sum = W.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum == 0, 1.0, row_sum)
        W /= row_sum
    return W


def build_inverse_distance_weight_matrix(
    coords: np.ndarray,
    power: float = 1.0,
    row_standardize: bool = True,
) -> np.ndarray:
    """W_ij = 1 / d_ij^power (inverse distance)."""
    dist = cdist(coords, coords, metric='euclidean')
    dist = np.where(dist < 1e-6, np.inf, dist)
    W = 1.0 / (dist ** power)
    np.fill_diagonal(W, 0.0)
    if row_standardize:
        row_sum = W.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum == 0, 1.0, row_sum)
        W /= row_sum
    return W


# ============================================================================
# Moran's I — Spatial Autocorrelation Diagnostic
# ============================================================================

@dataclass
class MoransIResult:
    """Moran's I test result for spatial autocorrelation."""

    I: float               # Moran's I statistic
    E_I: float             # expected value under null (-1/(n-1))
    var_I: float           # variance under null
    z_score: float         # standardised z-score
    p_value: float         # two-sided p-value (normal approx)
    significant: bool      # True if p < 0.05
    interpretation: str    # human-readable


def morans_i(
    y: np.ndarray,         # (n,) or (n_years, n) — variable of interest
    W: np.ndarray,         # (n,n) spatial weight matrix (row-standardised)
    standardise: bool = True,
) -> MoransIResult:
    """Moran's I test for spatial autocorrelation.

    I = (n / S0) * (zᵀ W z) / (zᵀ z)  where z = y - ȳ
    """
    if y.ndim == 2:
        y = y.mean(axis=0)  # average across years
    y_arr = np.asarray(y, float)
    W_mat = np.asarray(W, float)
    n = len(y_arr)
    z = y_arr - y_arr.mean()
    S0 = W_mat.sum()

    I_val = (n / S0) * (z @ W_mat @ z) / (z @ z) if S0 > 1e-12 else 0.0
    E_I = -1.0 / (n - 1)

    # Variance under normality assumption (Cliff & Ord 1981)
    S1_val = 0.5 * float(((W_mat + W_mat.T) ** 2).sum())
    S2_val = float(((W_mat.sum(axis=0) + W_mat.sum(axis=1)) ** 2).sum())
    s0_sq = float(S0 * S0)
    A = n * ((n * n - 3 * n + 3) * S1_val - n * S2_val + 3 * s0_sq)
    b2 = float((z ** 4).sum() / n / ((z ** 2).sum() / n) ** 2)  # kurtosis
    C = (n - 1) * (n - 2) * (n - 3) * s0_sq
    denom = float((n - 1) * (n - 2) * (n - 3) * s0_sq) if n > 3 else 1.0
    var_I = (A - b2 * C) / denom - E_I * E_I if denom > 0 else 1e-12
    var_I = max(float(var_I), 1e-12)
    z_score = (I_val - E_I) / np.sqrt(var_I)
    from scipy.stats import norm
    p_val = 2.0 * norm.sf(abs(z_score))
    sig = p_val < 0.05
    if sig:
        interp = ("显著空间正自相关" if I_val > E_I else "显著空间负自相关")
    else:
        interp = "无显著空间自相关（独立模糊集可能足够）"
    return MoransIResult(I=I_val, E_I=E_I, var_I=var_I, z_score=z_score,
                         p_value=p_val, significant=sig, interpretation=interp)


# ============================================================================
# Spatial Error Model (SEM) — covariance estimation
# ============================================================================

@dataclass
class SEMResult:
    """Spatial Error Model estimation results."""

    beta: np.ndarray          # regression coefficients (incl. intercept)
    lambda_spatial: float     # spatial autoregressive parameter for errors
    sigma2: float             # error variance σ²
    Cov: np.ndarray           # (n,n) spatial covariance = σ²[(I−λW)⁻¹(I−λW')⁻¹]
    log_likelihood: float
    converged: bool


def estimate_sem(
    y: np.ndarray,           # (T, n) panel — T years, n locations
    X: np.ndarray | None,    # (T, n, p) or (n, p) covariates
    W: np.ndarray,            # (n,n) spatial weight matrix (row-standardised)
) -> SEMResult:
    """Estimate Spatial Error Model via MLE.

    Model: y = Xβ + u,  u = λWu + ε,  ε ~ N(0, σ²I)
    Cov(u) = σ² · (I − λW)⁻¹ (I − λW')⁻¹

    For panel data y (T, n), we estimate on the time-average (between-city
    variation), which captures the long-run spatial correlation structure
    relevant for strategic facility-location decisions.
    """
    if y.ndim == 2:
        y_bar = y.mean(axis=0)  # (n,)
    else:
        y_bar = np.asarray(y, float)
    n = len(y_bar)

    # Build design matrix X (n, p+1) including intercept
    if X is None:
        X_design = np.ones((n, 1))
    elif X.ndim == 3:
        X_bar = X.mean(axis=0)  # (n, p)
        X_design = np.column_stack([np.ones(n), X_bar])
    else:
        X_design = np.column_stack([np.ones(n), np.asarray(X, float)])

    # OLS as initial guess
    XtX = X_design.T @ X_design
    Xty = X_design.T @ y_bar
    beta_ols = np.linalg.solve(XtX + np.eye(XtX.shape[0]) * 1e-10, Xty)
    resid_ols = y_bar - X_design @ beta_ols
    sigma2_ols = (resid_ols @ resid_ols) / n

    # MLE for λ (concentrated likelihood — profile over β, σ²)
    def neg_log_lik(lambda_val):
        if abs(lambda_val) >= 1.0:
            return 1e20
        # GLS transformation
        I = np.eye(n)
        A = I - lambda_val * W  # (I − λW)
        try:
            det_A = np.linalg.slogdet(A)[1]  # log|det(A)|
        except np.linalg.LinAlgError:
            return 1e20
        y_star = A @ y_bar
        X_star = A @ X_design
        XsX = X_star.T @ X_star
        Xsy = X_star.T @ y_star
        try:
            beta_gls = np.linalg.solve(XsX + np.eye(XsX.shape[0]) * 1e-10, Xsy)
        except np.linalg.LinAlgError:
            return 1e20
        resid = y_star - X_star @ beta_gls
        sigma2_gls = (resid @ resid) / n
        # Concentrated log-likelihood (negated for minimisation)
        nll = -det_A + 0.5 * n * np.log(max(sigma2_gls, 1e-12))
        return float(nll)

    res = minimize(neg_log_lik, x0=[0.3], bounds=[(-0.99, 0.99)], method='L-BFGS-B')

    lambda_hat = float(res.x[0])
    converged = res.success
    # Final GLS estimates
    I = np.eye(n)
    A = I - lambda_hat * W
    y_star = A @ y_bar
    X_star = A @ X_design
    XsX = X_star.T @ X_star
    Xsy = X_star.T @ y_star
    beta_hat = np.linalg.solve(XsX + np.eye(XsX.shape[0]) * 1e-10, Xsy)
    resid_final = y_star - X_star @ beta_hat
    sigma2_hat = float((resid_final @ resid_final) / n)

    # Spatial covariance matrix: Cov = σ² (I−λW)⁻¹ (I−λW')⁻¹
    # Compute via Neumann series or direct inverse
    try:
        A_inv = np.linalg.inv(A)
        Cov = sigma2_hat * (A_inv @ A_inv.T)
        Cov = 0.5 * (Cov + Cov.T)  # symmetrise
    except np.linalg.LinAlgError:
        # Fallback: Neumann series approximation
        Cov = sigma2_hat * np.eye(n)
        Ak = np.eye(n)
        for _ in range(20):
            Ak = Ak @ (lambda_hat * W)
            Cov += sigma2_hat * (Ak + Ak.T)
        Cov = 0.5 * (Cov + Cov.T)

    ll = float(-neg_log_lik(lambda_hat))
    return SEMResult(
        beta=beta_hat, lambda_spatial=lambda_hat, sigma2=sigma2_hat,
        Cov=Cov, log_likelihood=ll, converged=converged,
    )


# ============================================================================
# Hierarchical Σ — provincial macro + within-province micro
# ============================================================================

@dataclass
class HierarchicalSigma:
    """Two-level spatial covariance: Σ = Σ_macro + Σ_micro."""

    Sigma_macro: np.ndarray   # (n,n) provincial-level correlation
    Sigma_micro: np.ndarray   # (n,n) within-province residual correlation
    Sigma_total: np.ndarray   # Σ_macro + Σ_micro
    macro_var_share: float    # fraction of total variance from macro level
    province_labels: list[str]


def build_hierarchical_sigma(
    panel_data: np.ndarray,     # (T, n) — C&D waste generation (tons)
    city_names: list[str],
    province_labels: list[str],
    coords: np.ndarray,         # (n, 2) [lon, lat]
    k: int = 5,
) -> HierarchicalSigma:
    """Build two-level spatial covariance from panel data.

    Σ_macro: spatial correlation of provincial aggregates
    Σ_micro: within-province city-level residual correlation
    """
    n = len(city_names)
    # Group cities by province
    from collections import defaultdict
    prov_groups = defaultdict(list)
    for i, prov in enumerate(province_labels):
        prov_groups[prov].append(i)

    # Macro level: provincial aggregates
    n_prov = len(prov_groups)
    prov_agg = np.zeros((panel_data.shape[0], n_prov))
    prov_city_lists = list(prov_groups.values())
    prov_names = list(prov_groups.keys())
    for p_idx, city_idxs in enumerate(prov_city_lists):
        prov_agg[:, p_idx] = panel_data[:, city_idxs].sum(axis=1)

    # Macro correlation matrix → expand to city level
    if n_prov == 1:
        # All cities in the same province → no macro-level spatial structure
        Sigma_macro = np.zeros((n, n))
        macro_var_share = 0.0
    else:
        prov_corr = np.atleast_2d(np.corrcoef(prov_agg.T))  # (n_prov, n_prov)
        for i in range(n):
            pi = prov_names.index(province_labels[i])
            for j in range(n):
                pj = prov_names.index(province_labels[j])
                Sigma_macro[i, j] = prov_corr[pi, pj]
        macro_var_share = float(np.var(prov_agg)) / max(float(np.var(prov_agg)) + float(np.var(panel_detrended)), 1e-12)

    # Micro level: de-trend each city by its provincial mean, then fit spatial model
    panel_detrended = panel_data.copy()
    for p_name, city_idxs in prov_groups.items():
        if len(city_idxs) >= 2:
            for ci in city_idxs:
                panel_detrended[:, ci] -= prov_agg[:, prov_names.index(p_name)] / len(city_idxs)

    # Spatial Error Model on the detrended data (micro-level residual covariance)
    W_micro = build_knn_weight_matrix(coords, k=k)
    # Use a simple inverse-distance covariance for micro level
    dist = cdist(coords, coords, metric='euclidean') + 1e-6
    # For micro, use exponential decay within-province only
    Sigma_micro = np.zeros((n, n))
    for p_name, city_idxs in prov_groups.items():
        if len(city_idxs) >= 2:
            for ci in city_idxs:
                for cj in city_idxs:
                    if ci != cj:
                        Sigma_micro[ci, cj] = np.exp(-dist[ci, cj] / np.median(dist[dist > 0]))

    # Scale by micro-level variance
    if n_prov > 1:
        micro_var_total = float(np.var(panel_detrended))
        total_var = micro_var_total + float(np.var(prov_agg))
        if total_var > 1e-12:
            Sigma_micro *= micro_var_total / total_var
            Sigma_macro *= float(np.var(prov_agg)) / total_var

    mv_share = macro_var_share if n_prov == 1 else float(np.var(prov_agg)) / max(
        float(np.var(panel_detrended)) + float(np.var(prov_agg)), 1e-12)
    return HierarchicalSigma(
        Sigma_macro=Sigma_macro,
        Sigma_micro=Sigma_micro,
        Sigma_total=Sigma_macro + Sigma_micro,
        macro_var_share=mv_share,
        province_labels=province_labels,
    )


# ============================================================================
# Calibration comparison — naive vs. spatial-econometric Σ
# ============================================================================

@dataclass
class CalibrationCompare:
    """Compare naive (exp kernel) vs. SEM-based spatial covariance."""

    naive_min_eig: float
    sem_min_eig: float
    naive_condition: float
    sem_condition: float
    frobenius_diff: float
    spearman_corr: float       # rank correlation of Σ entries
    morans_i_MoranI: MoransIResult


def calibrate_and_compare(
    panel: np.ndarray,         # (T, n)
    coords: np.ndarray,        # (n, 2)
    city_names: list[str],
    province_labels: list[str],
    rho_naive: float = 100.0,
) -> CalibrationCompare:
    """Full pipeline: Moran's I → SEM → hierarchical Σ → comparison."""
    # Step 1: Moran's I diagnostic
    W = build_knn_weight_matrix(coords, k=5)
    y_bar = panel.mean(axis=0)
    mi = morans_i(y_bar, W)

    # Step 2: SEM estimation for comparison
    sem = estimate_sem(panel, X=None, W=W)

    # Step 3: Hierarchical Σ
    hier = build_hierarchical_sigma(panel, city_names, province_labels, coords)

    # Step 4: Naive Σ for comparison
    sigma_vec = panel.std(axis=0, ddof=1)
    dist_mat = cdist(coords, coords, metric='euclidean')
    from .covariance import build_spatial_sigma
    moments = Moments(mu=panel.mean(axis=0), sigma=sigma_vec)
    naive_sigma = build_spatial_sigma(moments, dist_mat, rho_km=rho_naive)

    # Metrics
    from .covariance import condition_number as cond, min_eigval as min_eig
    # Spearman rank correlation between entries
    from scipy.stats import spearmanr
    triu_naive = naive_sigma[np.triu_indices_from(naive_sigma, k=1)]
    triu_sem = sem.Cov[np.triu_indices_from(sem.Cov, k=1)]
    # Ensure same sign for correlation computation
    mask = np.isfinite(triu_naive) & np.isfinite(triu_sem)
    sp_corr = float(spearmanr(triu_naive[mask], triu_sem[mask])[0]) if mask.sum() > 3 else 0.0
    frob = float(np.linalg.norm(naive_sigma - sem.Cov, 'fro'))

    return CalibrationCompare(
        naive_min_eig=min_eig(naive_sigma),
        sem_min_eig=min_eig(sem.Cov),
        naive_condition=cond(naive_sigma),
        sem_condition=cond(sem.Cov),
        frobenius_diff=frob,
        spearman_corr=sp_corr,
        morans_i_MoranI=mi,
    )
