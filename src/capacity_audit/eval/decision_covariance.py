"""Decision-aligned diagnostics for multivariate covariance forecasts."""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.special import ndtr

from ..ambiguity.covariance import cholesky_sqrt
from ..models.dro_facility_location import FacilityLocationProblem
from ..models.recourse import solve_nominal_recourse


def energy_score_ensemble(
    samples: np.ndarray,
    observations: np.ndarray,
    independent_samples: np.ndarray,
) -> np.ndarray:
    """Return one energy score per multivariate observation."""
    draws = np.asarray(samples, float)
    draws_prime = np.asarray(independent_samples, float)
    observed = np.asarray(observations, float)
    if draws.ndim != 2 or draws_prime.shape != draws.shape:
        raise ValueError("samples and independent_samples must have equal 2D shape")
    if observed.ndim != 2 or observed.shape[1] != draws.shape[1]:
        raise ValueError("observations must have shape (T, dimension)")
    first = np.mean(
        np.linalg.norm(draws[:, None, :] - observed[None, :, :], axis=2),
        axis=0,
    )
    second = 0.5 * float(np.mean(np.linalg.norm(draws - draws_prime, axis=1)))
    return first - second


def variogram_score_ensemble(
    samples: np.ndarray,
    observations: np.ndarray,
    *,
    order: float = 0.5,
    weights: np.ndarray | None = None,
    pair_chunk: int = 64,
) -> np.ndarray:
    """Return normalized upper-triangle variogram scores.

    The scale normalization has no effect on rankings. It makes scores from
    different dimensions or weight matrices easier to compare.
    """
    draws = np.asarray(samples, float)
    observed = np.asarray(observations, float)
    if draws.ndim != 2 or observed.ndim != 2 or observed.shape[1] != draws.shape[1]:
        raise ValueError("samples and observations must share their second dimension")
    if order <= 0 or pair_chunk < 1:
        raise ValueError("order and pair_chunk must be positive")
    dimension = draws.shape[1]
    pair_i, pair_j = np.triu_indices(dimension, k=1)
    if weights is None:
        pair_weights = np.ones(len(pair_i), float)
    else:
        matrix = np.asarray(weights, float)
        if matrix.shape != (dimension, dimension) or np.any(matrix < 0):
            raise ValueError("weights must be a nonnegative square matrix")
        pair_weights = matrix[pair_i, pair_j]
    weight_sum = float(pair_weights.sum())
    if weight_sum <= 0:
        raise ValueError("at least one pair must have positive weight")
    pair_weights = pair_weights / weight_sum

    expected = np.empty(len(pair_i), float)
    for start in range(0, len(pair_i), pair_chunk):
        stop = min(start + pair_chunk, len(pair_i))
        differences = np.abs(
            draws[:, pair_i[start:stop]] - draws[:, pair_j[start:stop]]
        ) ** order
        expected[start:stop] = differences.mean(axis=0)
    realized = np.abs(
        observed[:, pair_i] - observed[:, pair_j]
    ) ** order
    return np.sum(
        pair_weights[None, :] * (realized - expected[None, :]) ** 2,
        axis=1,
    )


def gaussian_crps(
    observations: np.ndarray,
    means: np.ndarray,
    standard_deviations: np.ndarray,
) -> np.ndarray:
    """Closed-form CRPS for Gaussian forecasts, with lower values preferred."""
    observed, mean, scale = np.broadcast_arrays(
        np.asarray(observations, float),
        np.asarray(means, float),
        np.asarray(standard_deviations, float),
    )
    if np.any(scale <= 0) or not np.all(np.isfinite(scale)):
        raise ValueError("standard deviations must be positive and finite")
    z = (observed - mean) / scale
    density = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    return scale * (
        z * (2.0 * ndtr(z) - 1.0)
        + 2.0 * density
        - 1.0 / np.sqrt(np.pi)
    )


def projection_crps(
    covariance: np.ndarray,
    projections: np.ndarray,
    observations: np.ndarray,
    *,
    forecast_mean: np.ndarray | None = None,
) -> np.ndarray:
    """Score fixed one-dimensional projections of a Gaussian forecast.

    Returns an array with shape ``(n_observations, n_projections)``.
    """
    sigma = np.asarray(covariance, float)
    directions = np.asarray(projections, float)
    observed = np.asarray(observations, float)
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValueError("covariance must be square")
    dimension = sigma.shape[0]
    if directions.ndim != 2 or directions.shape[1] != dimension:
        raise ValueError("projections must have shape (R, dimension)")
    if observed.ndim != 2 or observed.shape[1] != dimension:
        raise ValueError("observations must have shape (T, dimension)")
    mean = np.zeros(dimension) if forecast_mean is None else np.asarray(forecast_mean, float)
    if mean.shape != (dimension,):
        raise ValueError("forecast_mean must have one entry per dimension")
    variances = np.einsum("ri,ij,rj->r", directions, sigma, directions)
    scales = np.sqrt(np.maximum(variances, 1e-15))
    projected_observations = observed @ directions.T
    projected_means = mean @ directions.T
    return gaussian_crps(
        projected_observations,
        projected_means[None, :],
        scales[None, :],
    )


def build_recourse_dual_projections(
    problem: FacilityLocationProblem,
    reference_decision: np.ndarray,
    forecast_means: np.ndarray,
    *,
    remove_common_mode: bool = False,
) -> np.ndarray:
    """Build unit-norm log-demand directions from fixed recourse duals."""
    decision = np.asarray(reference_decision, float)
    means = np.asarray(forecast_means, float)
    if decision.shape != (problem.n_facilities,):
        raise ValueError("reference_decision has the wrong dimension")
    if means.ndim != 2 or means.shape[1] != problem.n_customers:
        raise ValueError("forecast_means must have shape (T, n_customers)")
    directions = []
    for mean in means:
        recourse = solve_nominal_recourse(problem, decision, mean, backend="scipy")
        if recourse.status != "optimal" or not np.all(np.isfinite(recourse.pi_demand)):
            raise RuntimeError("recourse dual projection could not be computed")
        direction = recourse.pi_demand * mean
        if remove_common_mode:
            direction = direction - direction.mean()
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            raise RuntimeError("recourse dual produced a zero projection")
        directions.append(direction / norm)
    return np.asarray(directions)


def circular_block_bootstrap_indices(
    sample_size: int,
    block_length: int,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    """Generate circular moving-block bootstrap row indices."""
    if sample_size < 2 or not 1 <= block_length <= sample_size or n_bootstrap < 1:
        raise ValueError("invalid bootstrap dimensions")
    rng = np.random.default_rng(seed)
    blocks_needed = int(np.ceil(sample_size / block_length))
    starts = rng.integers(0, sample_size, size=(n_bootstrap, blocks_needed))
    offsets = np.arange(block_length)
    indices = (starts[:, :, None] + offsets[None, None, :]) % sample_size
    return indices.reshape(n_bootstrap, -1)[:, :sample_size]


def crossed_variation_attribution(
    decisions: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Decompose crossed binary-decision variation by exact sums of squares.

    The balanced outer-bootstrap x common-seed design has one observation per
    cell. Its observed variation separates exactly into outer, seed, and
    outer-by-seed interaction terms without an independence assumption.
    """
    values = np.asarray(decisions, float)
    if values.ndim != 3 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError("decisions must have shape (outer>=2, inner>=2, facilities)")
    if np.any((values != 0.0) & (values != 1.0)):
        raise ValueError("decisions must be binary")
    outer_count, seed_count, _ = values.shape
    grand_mean = values.mean(axis=(0, 1))
    outer_effect = values.mean(axis=1) - grand_mean
    seed_effect = values.mean(axis=0) - grand_mean
    interaction = (
        values
        - grand_mean[None, None, :]
        - outer_effect[:, None, :]
        - seed_effect[None, :, :]
    )
    outer_ss = seed_count * np.sum(outer_effect**2, axis=0)
    seed_ss = outer_count * np.sum(seed_effect**2, axis=0)
    interaction_ss = np.sum(interaction**2, axis=(0, 1))
    total_ss = np.sum(
        (values - grand_mean[None, None, :]) ** 2,
        axis=(0, 1),
    )
    closure_error = float(
        np.max(np.abs(outer_ss + seed_ss + interaction_ss - total_ss))
    )
    if closure_error > 1e-10:
        raise RuntimeError("crossed sums of squares do not close")
    aggregate_total = float(total_ss.sum())

    def share(component: np.ndarray) -> float:
        return float(component.sum() / aggregate_total) if aggregate_total > 0 else 0.0

    return {
        "outer_ss_by_facility": outer_ss,
        "seed_ss_by_facility": seed_ss,
        "interaction_ss_by_facility": interaction_ss,
        "total_ss_by_facility": total_ss,
        "outer_share": share(outer_ss),
        "seed_share": share(seed_ss),
        "interaction_share": share(interaction_ss),
        "closure_max_abs": closure_error,
    }


def crossed_hamming_summary(decisions: np.ndarray) -> dict[str, float]:
    """Compare outer-estimation and inner-SAA layout disagreement rates."""
    values = np.asarray(decisions, float)
    if values.ndim != 3:
        raise ValueError("decisions must have shape (outer, inner, facilities)")
    outer_count, inner_count, _ = values.shape
    estimation = [
        float(np.mean(values[left, seed] != values[right, seed]))
        for seed in range(inner_count)
        for left, right in combinations(range(outer_count), 2)
    ]
    saa = [
        float(np.mean(values[outer, left] != values[outer, right]))
        for outer in range(outer_count)
        for left, right in combinations(range(inner_count), 2)
    ]
    return {
        "estimation_mean_hamming": float(np.mean(estimation)) if estimation else 0.0,
        "estimation_max_hamming": float(np.max(estimation)) if estimation else 0.0,
        "saa_mean_hamming": float(np.mean(saa)) if saa else 0.0,
        "saa_max_hamming": float(np.max(saa)) if saa else 0.0,
    }


def generate_covariance_mixture_lognormal(
    forecast_means: np.ndarray,
    covariances: np.ndarray,
    n_scenarios: int,
    seed: int,
    standard_normals: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw equal-allocation scenarios from a bootstrap covariance mixture."""
    means = np.asarray(forecast_means, float)
    sigma = np.asarray(covariances, float)
    if means.ndim != 2 or np.any(means <= 0):
        raise ValueError("forecast_means must be a positive 2D array")
    if sigma.ndim != 3 or sigma.shape[1:] != (means.shape[1], means.shape[1]):
        raise ValueError("covariances must have shape (B, dimension, dimension)")
    if n_scenarios < sigma.shape[0]:
        raise ValueError("n_scenarios must cover every covariance at least once")
    rng = np.random.default_rng(seed)
    year_index = rng.integers(0, means.shape[0], size=n_scenarios)
    covariance_index = np.resize(np.arange(sigma.shape[0]), n_scenarios)
    rng.shuffle(covariance_index)
    if standard_normals is None:
        normals = rng.standard_normal((n_scenarios, means.shape[1]))
    else:
        normals = np.asarray(standard_normals, float)
        if normals.shape != (n_scenarios, means.shape[1]):
            raise ValueError("standard_normals has the wrong shape")
    scenarios = np.empty_like(normals)
    for bootstrap_index in range(sigma.shape[0]):
        rows = np.flatnonzero(covariance_index == bootstrap_index)
        covariance = sigma[bootstrap_index]
        root = cholesky_sqrt(covariance)
        shocks = normals[rows] @ root.T
        correction = -0.5 * np.diag(covariance)
        scenarios[rows] = means[year_index[rows]] * np.exp(shocks + correction)
    return scenarios, year_index, covariance_index
