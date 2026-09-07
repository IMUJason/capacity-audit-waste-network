"""Spatiotemporal calibration for annual city-level demand shocks.

The calibration operates on detrended log demand, preserving the panel time
dimension.  This avoids interpreting spatial clustering in long-run demand
levels as covariance of annual uncertainty.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.covariance import LedoitWolf

from .covariance import _psd_floor, cholesky_sqrt, rescale_to_sample_variances


@dataclass(frozen=True)
class LogTrendFit:
    intercept: np.ndarray
    slope: np.ndarray
    year_center: float
    residuals: np.ndarray
    fitted_log: np.ndarray
    smearing_factor: np.ndarray

    def forecast_arithmetic_mean(self, years: np.ndarray) -> np.ndarray:
        years_arr = np.asarray(years, float)
        log_median = (
            self.intercept[None, :]
            + (years_arr[:, None] - self.year_center) * self.slope[None, :]
        )
        return np.exp(log_median) * self.smearing_factor[None, :]


@dataclass(frozen=True)
class PooledSEMResult:
    lambda_spatial: float
    sigma2: float
    log_likelihood: float
    correlation: np.ndarray
    covariance: np.ndarray
    converged: bool


@dataclass(frozen=True)
class DampedTrendFit:
    residuals: np.ndarray
    fitted_log: np.ndarray
    forecast_log: np.ndarray
    forecast_mean: np.ndarray
    smearing_factor: np.ndarray


@dataclass(frozen=True)
class TrailingMeanFit:
    residuals: np.ndarray
    one_step_fitted_mean: np.ndarray
    forecast_mean: np.ndarray
    window: int


def fit_variance_matched_ledoit_wolf(residuals: np.ndarray) -> np.ndarray:
    """Estimate shrinkage correlation while preserving sample marginal variances."""
    errors = np.asarray(residuals, float)
    if errors.ndim != 2 or errors.shape[0] < 2:
        raise ValueError("residuals must contain at least two repeated cross-sections")
    sample_variances = errors.var(axis=0, ddof=1)
    raw = LedoitWolf(assume_centered=False).fit(errors).covariance_
    return rescale_to_sample_variances(raw, sample_variances)


def fit_log_linear_trends(panel: np.ndarray, years: np.ndarray) -> LogTrendFit:
    """Fit one log-linear trend per city and return annual log residuals."""
    values = np.asarray(panel, float)
    years_arr = np.asarray(years, float)
    if values.ndim != 2 or values.shape[0] != len(years_arr):
        raise ValueError("panel must have shape (n_years, n_cities)")
    if np.any(values <= 0):
        raise ValueError("log-linear calibration requires strictly positive demand")
    center = float(np.mean(years_arr))
    design = np.column_stack([np.ones(len(years_arr)), years_arr - center])
    coefficients, _, _, _ = np.linalg.lstsq(design, np.log(values), rcond=None)
    fitted = design @ coefficients
    residuals = np.log(values) - fitted
    smearing = np.mean(np.exp(residuals), axis=0)
    return LogTrendFit(
        intercept=coefficients[0],
        slope=coefficients[1],
        year_center=center,
        residuals=residuals,
        fitted_log=fitted,
        smearing_factor=smearing,
    )


def fit_damped_log_trends(panel: np.ndarray, n_forecast: int) -> DampedTrendFit:
    """Fit established damped-Holt trends to each city's log demand."""
    from statsmodels.tsa.holtwinters import Holt

    values = np.asarray(panel, float)
    if values.ndim != 2 or np.any(values <= 0) or n_forecast < 1:
        raise ValueError("positive panel and n_forecast >= 1 required")
    fitted = np.empty_like(values)
    forecast_log = np.empty((n_forecast, values.shape[1]))
    for city in range(values.shape[1]):
        model = Holt(
            np.log(values[:, city]),
            damped_trend=True,
            initialization_method="estimated",
        ).fit(optimized=True, remove_bias=False)
        fitted[:, city] = np.asarray(model.fittedvalues, float)
        forecast_log[:, city] = np.asarray(model.forecast(n_forecast), float)
    residuals = np.log(values) - fitted
    # The first fitted value is initialization-dominated and is excluded from
    # covariance estimation and Duan smearing.
    residuals = residuals[1:]
    smearing = np.mean(np.exp(residuals), axis=0)
    forecast_mean = np.exp(forecast_log) * smearing[None, :]
    return DampedTrendFit(
        residuals=residuals,
        fitted_log=fitted,
        forecast_log=forecast_log,
        forecast_mean=forecast_mean,
        smearing_factor=smearing,
    )


def fit_trailing_mean_forecast(
    panel: np.ndarray, n_forecast: int, window: int = 3
) -> TrailingMeanFit:
    """Forecast arithmetic demand with a trailing mean.

    Log forecast errors are genuine one-step-ahead errors: each fitted value
    uses only the preceding ``window`` observations. This is preferable to
    calibrating uncertainty from in-sample trend residuals when the covariance
    is intended to describe forecast uncertainty.
    """
    values = np.asarray(panel, float)
    if (
        values.ndim != 2
        or np.any(values <= 0)
        or n_forecast < 1
        or window < 1
        or values.shape[0] <= window
    ):
        raise ValueError(
            "positive panel, n_forecast >= 1, and 1 <= window < n_years required"
        )
    fitted = np.vstack(
        [values[index - window:index].mean(axis=0) for index in range(window, len(values))]
    )
    residuals = np.log(values[window:]) - np.log(fitted)
    terminal_mean = values[-window:].mean(axis=0)
    forecast_mean = np.repeat(terminal_mean[None, :], n_forecast, axis=0)
    return TrailingMeanFit(
        residuals=residuals,
        one_step_fitted_mean=fitted,
        forecast_mean=forecast_mean,
        window=window,
    )


def fit_pooled_sem(residuals: np.ndarray, weights: np.ndarray) -> PooledSEMResult:
    """Fit a pooled SEM to repeated annual cross-sections of log residuals.

    For residual row ``e_t``, ``(I-lambda W)e_t = innovation_t``.  The
    likelihood contains one Jacobian contribution per year.
    """
    errors = np.asarray(residuals, float)
    W = np.asarray(weights, float)
    if errors.ndim != 2 or W.shape != (errors.shape[1], errors.shape[1]):
        raise ValueError("residual and weight-matrix dimensions do not match")
    errors = errors - errors.mean(axis=0, keepdims=True)
    t_count, n = errors.shape
    identity = np.eye(n)

    def profile_nll(value: float) -> float:
        A = identity - float(value) * W
        sign, logdet = np.linalg.slogdet(A)
        if sign == 0 or not np.isfinite(logdet):
            return 1e100
        innovations = errors @ A.T
        sigma2 = float(np.sum(innovations**2) / (t_count * n))
        return float(-t_count * logdet + 0.5 * t_count * n * np.log(max(sigma2, 1e-15)))

    result = minimize_scalar(profile_nll, bounds=(-0.95, 0.95), method="bounded")
    lam = float(result.x)
    A_inv = np.linalg.inv(identity - lam * W)
    innovations = errors @ (identity - lam * W).T
    sigma2 = float(np.sum(innovations**2) / (t_count * n))
    raw = sigma2 * A_inv @ A_inv.T
    raw = 0.5 * (raw + raw.T)
    raw_std = np.sqrt(np.maximum(np.diag(raw), 1e-15))
    correlation = raw / np.outer(raw_std, raw_std)
    correlation = _psd_floor(correlation, 1e-12)
    corr_std = np.sqrt(np.maximum(np.diag(correlation), 1e-15))
    correlation = correlation / np.outer(corr_std, corr_std)
    np.fill_diagonal(correlation, 1.0)
    target_variance = np.var(errors, axis=0, ddof=1)
    covariance = rescale_to_sample_variances(correlation, target_variance)
    return PooledSEMResult(
        lambda_spatial=lam,
        sigma2=sigma2,
        log_likelihood=-float(result.fun),
        correlation=correlation,
        covariance=covariance,
        converged=bool(result.success),
    )


def pooled_morans_i(residuals: np.ndarray, weights: np.ndarray) -> float:
    """Mean annual Moran's I for detrended residual cross-sections."""
    errors = np.asarray(residuals, float)
    W = np.asarray(weights, float)
    n = errors.shape[1]
    s0 = float(W.sum())
    values = []
    for row in errors:
        z = row - row.mean()
        denominator = float(z @ z)
        if denominator > 1e-15:
            values.append((n / s0) * float(z @ W @ z) / denominator)
    return float(np.mean(values))


def pooled_moran_permutation_test(
    residuals: np.ndarray,
    weights: np.ndarray,
    n_permutations: int = 9_999,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    """Upper-tail spatial-label randomization test for mean annual Moran's I."""
    observed = pooled_morans_i(residuals, weights)
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations)
    for index in range(n_permutations):
        permutation = rng.permutation(residuals.shape[1])
        null[index] = pooled_morans_i(residuals[:, permutation], weights)
    p_value = float((np.count_nonzero(null >= observed) + 1) / (n_permutations + 1))
    return observed, float(null.mean()), float(null.std(ddof=1)), p_value


def bootstrap_sem_lambda(
    residuals: np.ndarray,
    weights: np.ndarray,
    n_bootstrap: int = 500,
    seed: int = 123,
) -> tuple[float, float]:
    """Percentile interval from resampling annual residual cross-sections."""
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_bootstrap)
    for index in range(n_bootstrap):
        sample = residuals[rng.integers(0, residuals.shape[0], residuals.shape[0])]
        estimates[index] = fit_pooled_sem(sample, weights).lambda_spatial
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return float(lower), float(upper)


def block_bootstrap_sem_lambda(
    residuals: np.ndarray,
    weights: np.ndarray,
    block_length: int = 3,
    n_bootstrap: int = 500,
    seed: int = 123,
) -> tuple[float, float]:
    """Circular moving-block bootstrap interval for pooled SEM lambda."""
    errors = np.asarray(residuals, float)
    if errors.ndim != 2 or not 1 <= block_length <= errors.shape[0]:
        raise ValueError("block_length must be between 1 and the number of years")
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_bootstrap)
    blocks_needed = int(np.ceil(errors.shape[0] / block_length))
    offsets = np.arange(block_length)
    for index in range(n_bootstrap):
        starts = rng.integers(0, errors.shape[0], size=blocks_needed)
        sample_index = np.concatenate(
            [(start + offsets) % errors.shape[0] for start in starts]
        )[:errors.shape[0]]
        estimates[index] = fit_pooled_sem(
            errors[sample_index], weights
        ).lambda_spatial
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return float(lower), float(upper)


def generate_lognormal_mixture_scenarios(
    forecast_means: np.ndarray,
    log_covariance: np.ndarray,
    n_scenarios: int,
    seed: int,
    standard_normals: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate annual-demand scenarios with a mixture over forecast years."""
    means = np.asarray(forecast_means, float)
    covariance = _psd_floor(np.asarray(log_covariance, float), 1e-12)
    rng = np.random.default_rng(seed)
    year_index = np.arange(n_scenarios) % means.shape[0]
    rng.shuffle(year_index)
    if standard_normals is None:
        standard_normals = rng.standard_normal((n_scenarios, means.shape[1]))
    normals = np.asarray(standard_normals, float)
    if normals.shape != (n_scenarios, means.shape[1]):
        raise ValueError("standard_normals has the wrong shape")
    root = cholesky_sqrt(covariance)
    log_mean = np.log(np.maximum(means[year_index], 1e-12)) - 0.5 * np.diag(covariance)
    scenarios = np.exp(log_mean + normals @ root.T)
    return scenarios, year_index, normals
