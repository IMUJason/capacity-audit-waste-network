"""Decision-dependent spatial covariance Σ(y).

    σ_i(y) = σ_i₀ exp(−β · n_i(y)),   n_i(y) = Σ_j y_j · I(d_ij < R)
    ρ(y)   = ρ₀ · (1 + γ · Σ_j y_j)

This module captures the bidirectional coupling: facility decisions change the
spatial uncertainty landscape. When β > 0, opening facilities near a city
reduces its demand uncertainty (better logistics visibility, more data).
"""
from __future__ import annotations

import numpy as np

from ..data.schema import Moments, Instance
from ..models.dro_facility_location import FacilityLocationProblem
from .ambiguity_set import AmbiguitySet
from .covariance import haversine_km_matrix, build_spatial_sigma
from ..solvers.cplex_misocp import solve_deterministic_nominal, MISOCPResult


def compute_neighbourhood_counts(
    y_open: np.ndarray,
    dist_km: np.ndarray,   # (n_cust, n_fac)
    radius_km: float = 80.0,
) -> np.ndarray:
    """n_i(y) = number of open facilities within radius of each city."""
    return (np.asarray(dist_km, float) < radius_km) @ np.asarray(y_open > 0.5, float)


def decision_dependent_sigma(
    base_moments: Moments,
    dist_cust_fac: np.ndarray,  # (n, m)
    dist_cust_cust: np.ndarray, # (n, n)  haversine between cities
    y_open: np.ndarray,         # (m,)
    beta: float,                # uncertainty-reduction strength
    radius_km: float,           # influence radius (km)
    gamma_rho: float = 0.0,     # ρ expansion per facility
    base_rho_km: float = 100.0,
) -> tuple[Moments, float]:
    """Return (decision-dependent Moments, ρ(y)) for a given facility selection."""
    mu = np.asarray(base_moments.mu, float).copy()
    sigma_base = np.asarray(base_moments.sigma, float)
    n_y = compute_neighbourhood_counts(y_open, dist_cust_fac, radius_km)
    sigma_y = sigma_base * np.exp(-beta * n_y)
    sigma_y = np.maximum(sigma_y, sigma_base * 0.3)  # floor at 30% of base
    rho_y = base_rho_km * (1.0 + gamma_rho * np.sum(y_open > 0.5))
    from .covariance import haversine_km_matrix
    return Moments(mu=mu, sigma=sigma_y), max(rho_y, 5.0)


def solve_decision_dependent_dro(
    problem: FacilityLocationProblem,
    dist_cust_fac: np.ndarray,
    dist_cust_cust: np.ndarray,
    base_moments: Moments,
    gamma1: float, gamma2: float,
    beta: float = 0.01,
    radius_km: float = 80.0,
    max_iter: int = 10,
    time_limit: int = 300,
) -> tuple[MISOCPResult, list[dict]]:
    """Fixed-point iteration for decision-dependent Σ(y).

    1. Start with y₀ = decision-independent solution.
    2. Given y_k, compute Σ(y_k) → solve DRO → y_{k+1}.
    3. Repeat until convergence or max_iter.
    """
    import time
    history = []
    # Initial: decision-independent
    mom0 = base_moments
    rho0 = float(np.median(np.sort(dist_cust_cust, axis=1)[:, 1:6]))
    amb0 = AmbiguitySet.from_moments(
        mom0, kind="joint_spatial", gamma1=gamma1, gamma2=gamma2,
        haversine_km=dist_cust_cust, rho_km=rho0)
    kappa0 = amb0.omega
    robust_d = amb0.mu + kappa0 * np.sqrt(np.diag(amb0.sigma_matrix))
    t_start = time.time()
    res_prev = solve_deterministic_nominal(problem, robust_d, time_limit=time_limit)

    y_best = res_prev.y_open.copy()
    obj_best = res_prev.objective
    converged = False

    history.append({"iter": 0, "y": np.where(y_best > 0.5)[0].tolist(),
                    "obj": obj_best, "beta_eff": 0, "converged": False})

    for it in range(1, max_iter + 1):
        # Compute Σ(y_{k-1})
        mom_y, rho_y = decision_dependent_sigma(
            base_moments, dist_cust_fac, dist_cust_cust,
            y_best, beta=beta, radius_km=radius_km, base_rho_km=rho0)
        amb_y = AmbiguitySet.from_moments(
            mom_y, kind="joint_spatial", gamma1=gamma1, gamma2=gamma2,
            haversine_km=dist_cust_cust, rho_km=rho_y)
        kappa_y = amb_y.omega
        robust_d_y = amb_y.mu + kappa_y * np.sqrt(np.diag(amb_y.sigma_matrix))
        res_new = solve_deterministic_nominal(problem, robust_d_y, time_limit=time_limit)

        y_new = res_new.y_open.copy()
        obj_new = res_new.objective
        changed = not np.array_equal(y_new > 0.5, y_best > 0.5)

        history.append({"iter": it,
                        "y": np.where(y_new > 0.5)[0].tolist(),
                        "obj": obj_new, "changed": changed,
                        "mean_sigma_ratio": float(np.mean(mom_y.sigma / (base_moments.sigma + 1e-9))),
                        "rho_y": rho_y})

        if changed and obj_new < obj_best:
            y_best, obj_best = y_new, obj_new

        if not changed:
            converged = True
            history[-1]["converged"] = True
            break

    result = MISOCPResult(
        y_open=y_best, objective=obj_best,
        solve_time=time.time() - t_start,
        status="converged" if converged else "max_iter",
        extra={"history": history},
    )
    return result, history
