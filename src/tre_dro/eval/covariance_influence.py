"""Local covariance influence for fixed mean-CVaR facility layouts.

The derivatives in this module are restricted to covariance perturbations with
zero diagonal.  This keeps every lognormal marginal distribution fixed while
changing only cross-city dependence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigvalsh, solve_sylvester

from ..ambiguity.covariance import cholesky_sqrt
from ..models.dro_facility_location import FacilityLocationProblem
from ..models.recourse import solve_nominal_recourse


@dataclass(frozen=True)
class RecourseBatch:
    costs: np.ndarray
    demand_duals: np.ndarray | None


@dataclass(frozen=True)
class MeanCVaRInfluence:
    covariance_gradient: np.ndarray
    correlation_gradient: np.ndarray
    var_threshold: float
    tail_weights: np.ndarray
    scenario_weights: np.ndarray


def empirical_cvar_weights(losses: np.ndarray, alpha: float) -> tuple[float, np.ndarray]:
    """Return a VaR threshold and empirical CVaR subgradient weights.

    The returned nonnegative weights sum to one and reproduce the
    Rockafellar-Uryasev empirical CVaR, including the fractional mass required
    at the empirical VaR when ``(1-alpha) * n`` is not an integer.
    """
    values = np.asarray(losses, float)
    if (
        values.ndim != 1
        or len(values) == 0
        or not np.all(np.isfinite(values))
        or not 0.0 < alpha < 1.0
    ):
        raise ValueError("losses must be a nonempty finite vector and alpha valid")
    threshold = float(np.quantile(values, alpha, method="higher"))
    cap = 1.0 / ((1.0 - alpha) * len(values))
    weights = np.zeros(len(values), float)
    above = values > threshold
    weights[above] = cap
    remaining = 1.0 - float(weights.sum())
    at_threshold = np.flatnonzero(values == threshold)
    if remaining < -1e-10 or len(at_threshold) == 0:
        raise RuntimeError("could not construct empirical CVaR weights")
    weights[at_threshold] = max(0.0, remaining) / len(at_threshold)
    if not np.isclose(weights.sum(), 1.0, atol=1e-10):
        raise RuntimeError("empirical CVaR weights do not sum to one")
    if np.any(weights < -1e-12) or np.any(weights > cap + 1e-10):
        raise RuntimeError("invalid empirical CVaR subgradient weights")
    return threshold, weights


def empirical_mean_cvar(
    losses: np.ndarray, *, risk_weight: float, cvar_alpha: float
) -> float:
    """Evaluate the empirical mean-CVaR functional used by the model."""
    values = np.asarray(losses, float)
    if not 0.0 <= risk_weight <= 1.0:
        raise ValueError("risk_weight must lie in [0, 1]")
    _, tail_weights = empirical_cvar_weights(values, cvar_alpha)
    return float(
        (1.0 - risk_weight) * values.mean()
        + risk_weight * tail_weights @ values
    )


def fixed_diagonal_mean_cvar_influence(
    covariance: np.ndarray,
    standard_normals: np.ndarray,
    demands: np.ndarray,
    recourse_costs: np.ndarray,
    demand_duals: np.ndarray,
    *,
    risk_weight: float,
    cvar_alpha: float,
) -> MeanCVaRInfluence:
    """Estimate the pathwise covariance gradient for one fixed layout.

    For ``D = mu * exp(L z - diag(Sigma)/2)`` and a zero-diagonal covariance
    direction ``H``, ``dL`` solves ``L dL + dL L = H``.  The recourse LP is
    differentiable almost surely under continuous lognormal demand, with
    demand gradient equal to its demand dual.  CVaR contributes its empirical
    tail subgradient weights.

    Diagonal entries of the returned covariance gradient are intentionally not
    interpretable because the lognormal mean correction is held fixed.  The
    correlation gradient is projected to a zero diagonal before return.
    """
    sigma = np.asarray(covariance, float)
    normals = np.asarray(standard_normals, float)
    demand = np.asarray(demands, float)
    costs = np.asarray(recourse_costs, float)
    duals = np.asarray(demand_duals, float)
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValueError("covariance must be square")
    scenarios, dimension = normals.shape
    if (
        demand.shape != (scenarios, dimension)
        or duals.shape != demand.shape
        or costs.shape != (scenarios,)
        or sigma.shape != (dimension, dimension)
        or not np.all(np.isfinite(demand))
        or not np.all(np.isfinite(duals))
        or not np.all(np.isfinite(costs))
    ):
        raise ValueError("scenario arrays have incompatible shapes or values")
    if not 0.0 <= risk_weight <= 1.0:
        raise ValueError("risk_weight must lie in [0, 1]")

    threshold, tail_weights = empirical_cvar_weights(costs, cvar_alpha)
    scenario_weights = (
        np.full(scenarios, (1.0 - risk_weight) / scenarios)
        + risk_weight * tail_weights
    )
    demand_subgradients = demand * duals
    root_pullback = np.einsum(
        "s,si,sj->ij",
        scenario_weights,
        demand_subgradients,
        normals,
        optimize=True,
    )
    root_pullback = 0.5 * (root_pullback + root_pullback.T)
    root = cholesky_sqrt(sigma)
    covariance_gradient = solve_sylvester(root, root, root_pullback)
    covariance_gradient = 0.5 * (covariance_gradient + covariance_gradient.T)

    standard_deviations = np.sqrt(np.maximum(np.diag(sigma), 0.0))
    if np.any(standard_deviations <= 0.0):
        raise ValueError("covariance must have a strictly positive diagonal")
    correlation_gradient = (
        standard_deviations[:, None]
        * covariance_gradient
        * standard_deviations[None, :]
    )
    correlation_gradient = 0.5 * (
        correlation_gradient + correlation_gradient.T
    )
    np.fill_diagonal(correlation_gradient, 0.0)
    return MeanCVaRInfluence(
        covariance_gradient=covariance_gradient,
        correlation_gradient=correlation_gradient,
        var_threshold=threshold,
        tail_weights=tail_weights,
        scenario_weights=scenario_weights,
    )


def covariance_to_correlation(covariance: np.ndarray) -> np.ndarray:
    """Convert a positive-diagonal covariance matrix to a correlation matrix."""
    sigma = np.asarray(covariance, float)
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValueError("covariance must be square")
    standard_deviations = np.sqrt(np.maximum(np.diag(sigma), 0.0))
    if np.any(standard_deviations <= 0.0):
        raise ValueError("covariance must have a strictly positive diagonal")
    correlation = sigma / np.outer(standard_deviations, standard_deviations)
    correlation = 0.5 * (correlation + correlation.T)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def correlation_to_covariance(
    correlation: np.ndarray, marginal_variances: np.ndarray
) -> np.ndarray:
    """Restore a covariance matrix while preserving supplied marginal variances."""
    matrix = np.asarray(correlation, float)
    variances = np.asarray(marginal_variances, float)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or variances.shape != (matrix.shape[0],)
        or np.any(variances <= 0.0)
    ):
        raise ValueError("correlation and marginal variances are incompatible")
    standard_deviations = np.sqrt(variances)
    covariance = standard_deviations[:, None] * matrix * standard_deviations[None, :]
    return 0.5 * (covariance + covariance.T)


def steepest_fixed_diagonal_direction(
    objective_gap: float, correlation_gradient: np.ndarray
) -> tuple[np.ndarray, float]:
    """Return the unit-Frobenius direction that closes a linearized gap fastest."""
    gradient = np.asarray(correlation_gradient, float).copy()
    if gradient.ndim != 2 or gradient.shape[0] != gradient.shape[1]:
        raise ValueError("correlation_gradient must be square")
    gradient = 0.5 * (gradient + gradient.T)
    np.fill_diagonal(gradient, 0.0)
    norm = float(np.linalg.norm(gradient, ord="fro"))
    if not np.isfinite(objective_gap) or abs(objective_gap) <= 0.0 or norm <= 0.0:
        raise ValueError("a nonzero finite gap and gradient are required")
    direction = -np.sign(objective_gap) * gradient / norm
    return direction, float(abs(objective_gap) / norm)


def maximum_psd_step(correlation: np.ndarray, direction: np.ndarray) -> float:
    """Maximum nonnegative step before ``correlation + t*direction`` loses PSD."""
    matrix = np.asarray(correlation, float)
    tangent = np.asarray(direction, float)
    if matrix.shape != tangent.shape or matrix.ndim != 2:
        raise ValueError("correlation and direction must be equal square matrices")
    matrix = 0.5 * (matrix + matrix.T)
    tangent = 0.5 * (tangent + tangent.T)
    if float(np.linalg.eigvalsh(matrix).min()) <= 0.0:
        raise ValueError("correlation must be positive definite")
    generalized = eigvalsh(tangent, matrix)
    minimum = float(generalized.min())
    return float(-1.0 / minimum) if minimum < 0.0 else float("inf")


def evaluate_recourse_batch(
    problem: FacilityLocationProblem,
    decision: np.ndarray,
    scenarios: np.ndarray,
    *,
    backend: str = "cplex",
    return_duals: bool = True,
) -> RecourseBatch:
    """Evaluate fixed-layout recourse costs and optional demand duals in a batch."""
    demand = np.asarray(scenarios, float)
    layout = np.asarray(decision, float)
    if demand.ndim != 2 or demand.shape[1] != problem.n_customers:
        raise ValueError("scenarios have the wrong shape")
    if layout.shape != (problem.n_facilities,):
        raise ValueError("decision has the wrong shape")
    if backend == "scipy":
        costs = np.empty(len(demand), float)
        duals = np.empty_like(demand) if return_duals else None
        for index, row in enumerate(demand):
            result = solve_nominal_recourse(problem, layout, row, backend="scipy")
            if result.status != "optimal":
                raise RuntimeError(f"recourse LP failed for scenario {index}")
            costs[index] = result.total_cost
            if duals is not None:
                duals[index] = result.pi_demand
        return RecourseBatch(costs=costs, demand_duals=duals)
    if backend != "cplex":
        raise ValueError("backend must be 'cplex' or 'scipy'")
    return _evaluate_recourse_cplex(
        problem, layout, demand, return_duals=return_duals
    )


def _evaluate_recourse_cplex(
    problem: FacilityLocationProblem,
    decision: np.ndarray,
    scenarios: np.ndarray,
    *,
    return_duals: bool,
) -> RecourseBatch:
    import cplex

    open_ids = np.flatnonzero(decision > 0.5)
    if len(open_ids) == 0:
        raise ValueError("at least one facility must be open")
    n = problem.n_customers
    n_open = len(open_ids)
    n_flow = n * n_open
    unit_cost = (
        problem.cost_per_ton_km * problem.distance_km[:, open_ids]
        + problem.processing_costs[open_ids][None, :]
    )
    objective = np.concatenate([unit_cost.reshape(-1), np.full(n, problem.unmet_penalty)])

    model = cplex.Cplex()
    model.set_log_stream(None)
    model.set_error_stream(None)
    model.set_warning_stream(None)
    model.set_results_stream(None)
    model.objective.set_sense(model.objective.sense.minimize)
    model.variables.add(obj=objective.tolist(), lb=[0.0] * len(objective))
    rows = []
    senses = []
    rhs = []
    for customer in range(n):
        indices = list(range(customer * n_open, (customer + 1) * n_open))
        indices.append(n_flow + customer)
        rows.append(cplex.SparsePair(ind=indices, val=[1.0] * len(indices)))
        senses.append("E")
        rhs.append(float(scenarios[0, customer]))
    for facility, facility_id in enumerate(open_ids):
        indices = list(range(facility, n_flow, n_open))
        rows.append(cplex.SparsePair(ind=indices, val=[1.0] * len(indices)))
        senses.append("L")
        rhs.append(float(problem.capacities[facility_id]))
    model.linear_constraints.add(lin_expr=rows, senses=senses, rhs=rhs)
    model.parameters.lpmethod.set(model.parameters.lpmethod.values.dual)
    model.parameters.advance.set(1)

    costs = np.empty(len(scenarios), float)
    duals = np.empty_like(scenarios) if return_duals else None
    for index, row in enumerate(scenarios):
        if index:
            model.linear_constraints.set_rhs(
                [(customer, float(row[customer])) for customer in range(n)]
            )
        model.solve()
        if not model.solution.is_primal_feasible():
            model.end()
            raise RuntimeError(f"CPLEX recourse LP failed for scenario {index}")
        costs[index] = float(model.solution.get_objective_value())
        if duals is not None:
            duals[index] = np.asarray(
                model.solution.get_dual_values(list(range(n))), float
            )
    model.end()
    return RecourseBatch(costs=costs, demand_duals=duals)
