"""Decision-complete scoring and stability diagnostics for recourse costs.

The score evaluates the push-forward distribution of recourse cost for every
layout in a set fixed before holdout evaluation.  It is therefore proper for
those cost distributions, not for the full multivariate demand distribution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..models.dro_facility_location import FacilityLocationProblem


def _finite_vector(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, float)
    if array.ndim != 1 or len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a nonempty finite vector")
    return array


def mean_pairwise_absolute(values: np.ndarray) -> float:
    """Return E|X-X'| for the equally weighted empirical distribution."""
    ordered = np.sort(_finite_vector(values, "values"))
    count = len(ordered)
    coefficients = 2.0 * np.arange(count) - count + 1.0
    return float(2.0 * coefficients @ ordered / count**2)


def mean_cross_absolute(left: np.ndarray, right: np.ndarray) -> float:
    """Return E|X-Y| for two equally weighted empirical distributions."""
    x = _finite_vector(left, "left")
    y = np.sort(_finite_vector(right, "right"))
    prefix = np.concatenate([[0.0], np.cumsum(y)])
    positions = np.searchsorted(y, x, side="right")
    lower = x * positions - prefix[positions]
    upper = (prefix[-1] - prefix[positions]) - x * (len(y) - positions)
    return float(np.sum(lower + upper) / (len(x) * len(y)))


def ensemble_crps(
    samples: np.ndarray, observations: np.ndarray, *, fair: bool = False
) -> np.ndarray:
    """Return ensemble CRPS for each scalar observation.

    ``fair=True`` removes the finite-ensemble bias when the members are Monte
    Carlo draws from an underlying forecast distribution.  The default scores
    the equally weighted empirical distribution itself and is the version that
    satisfies the exact empirical CRPS-divergence identity.
    """
    draws = _finite_vector(samples, "samples")
    observed = _finite_vector(observations, "observations")
    pairwise = mean_pairwise_absolute(draws)
    if fair:
        if len(draws) < 2:
            raise ValueError("fair ensemble CRPS requires at least two samples")
        pairwise *= len(draws) / (len(draws) - 1.0)
    correction = 0.5 * pairwise
    return np.mean(np.abs(draws[:, None] - observed[None, :]), axis=0) - correction


def crps_divergence(left: np.ndarray, right: np.ndarray) -> float:
    """Return the CRPS divergence, integral (F-G)^2 dt, for two ensembles."""
    value = (
        mean_cross_absolute(left, right)
        - 0.5 * mean_pairwise_absolute(left)
        - 0.5 * mean_pairwise_absolute(right)
    )
    return float(max(0.0, value))


def empirical_cvar(values: np.ndarray, alpha: float) -> float:
    """Evaluate upper-tail CVaR through its Rockafellar-Uryasev formula."""
    losses = _finite_vector(values, "values")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    eta = float(np.quantile(losses, alpha, method="higher"))
    return float(eta + np.mean(np.maximum(losses - eta, 0.0)) / (1.0 - alpha))


def mean_cvar_objectives(
    recourse_costs: np.ndarray,
    fixed_costs: np.ndarray,
    *,
    risk_weight: float,
    cvar_alpha: float,
) -> np.ndarray:
    """Evaluate one fixed-plus-mean-CVaR objective per candidate layout."""
    costs = np.asarray(recourse_costs, float)
    fixed = np.asarray(fixed_costs, float)
    if (
        costs.ndim != 2
        or costs.shape[0] != len(fixed)
        or costs.shape[1] == 0
        or not np.all(np.isfinite(costs))
        or not np.all(np.isfinite(fixed))
    ):
        raise ValueError("costs must be finite (layouts, scenarios) data")
    if not 0.0 <= risk_weight <= 1.0 or not 0.0 < cvar_alpha < 1.0:
        raise ValueError("invalid risk parameters")
    cvars = np.array([empirical_cvar(row, cvar_alpha) for row in costs])
    return fixed + (1.0 - risk_weight) * costs.mean(axis=1) + risk_weight * cvars


def mean_cvar_lipschitz_factor(risk_weight: float, cvar_alpha: float) -> float:
    """Lipschitz factor of mean-CVaR with respect to Wasserstein-1 distance."""
    if not 0.0 <= risk_weight <= 1.0 or not 0.0 < cvar_alpha < 1.0:
        raise ValueError("invalid risk parameters")
    return float((1.0 - risk_weight) + risk_weight / (1.0 - cvar_alpha))


def crps_to_wasserstein_bound(
    divergence: float,
    threshold: float,
    *,
    forecast_tail_excess: float = 0.0,
    reference_tail_excess: float = 0.0,
) -> float:
    """Bound W1 using CRPS divergence on [0, threshold] plus tail excess.

    For nonnegative costs X and Y,

        W1(F_X,F_Y) <= sqrt(M * integral(F_X-F_Y)^2)
                       + E[(X-M)+] + E[(Y-M)+].

    The tail terms are indispensable for unbounded cost distributions.
    """
    values = np.array(
        [divergence, threshold, forecast_tail_excess, reference_tail_excess], float
    )
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or threshold <= 0.0:
        raise ValueError("bound inputs must be finite and nonnegative, with M > 0")
    return float(
        np.sqrt(threshold * divergence)
        + forecast_tail_excess
        + reference_tail_excess
    )


def aggregate_crps_objective_bound(
    aggregate_divergence: float,
    weights: np.ndarray,
    support_upper_bounds: np.ndarray,
    *,
    risk_weight: float,
    cvar_alpha: float,
) -> float:
    """Uniform objective-error bound from aggregate CRPS regret.

    This bounded-support version uses D_y <= R / w_y.  It is intentionally
    separate from the tail-aware layout-wise bound so callers cannot silently
    apply a compact-support theorem to lognormal costs.
    """
    weight = np.asarray(weights, float)
    upper = np.asarray(support_upper_bounds, float)
    if (
        weight.ndim != 1
        or upper.shape != weight.shape
        or len(weight) == 0
        or not np.all(np.isfinite(weight))
        or not np.all(np.isfinite(upper))
        or np.any(weight <= 0.0)
        or np.any(upper <= 0.0)
        or not np.isfinite(aggregate_divergence)
        or aggregate_divergence < 0.0
    ):
        raise ValueError("positive weights, support bounds, and regret are required")
    normalized = weight / weight.sum()
    w1_bound = np.max(np.sqrt(upper * aggregate_divergence / normalized))
    return float(
        mean_cvar_lipschitz_factor(risk_weight, cvar_alpha) * w1_bound
    )


@dataclass(frozen=True)
class DecisionMarginCertificate:
    reference_best: int
    comparison_best: int
    reference_gap: float
    uniform_error: float
    gap_condition_holds: bool
    optimizer_agrees: bool


@dataclass(frozen=True)
class EmpiricalTailBound:
    wasserstein_bound: float
    threshold: float
    divergence: float
    forecast_tail_excess: float
    reference_tail_excess: float


@dataclass(frozen=True)
class AdjacentLayoutPair:
    closed_layout: int
    open_layout: int
    added_facility: int


@dataclass(frozen=True)
class MarginalBenefit:
    expected_recourse_saving: float
    cvar_recourse_saving: float
    gross_risk_adjusted_benefit: float
    annual_fixed_cost: float
    net_opening_benefit: float


def adjacent_layout_pairs(layouts: np.ndarray) -> tuple[AdjacentLayoutPair, ...]:
    """Enumerate every nested candidate pair that differs by one facility."""
    candidates = np.asarray(layouts, int)
    if candidates.ndim != 2 or len(candidates) < 2 or np.any(
        (candidates != 0) & (candidates != 1)
    ):
        raise ValueError("layouts must be a binary matrix with at least two rows")
    if len({tuple(row) for row in candidates}) != len(candidates):
        raise ValueError("candidate layouts must be unique")
    pairs = []
    for closed_index, closed in enumerate(candidates):
        for open_index, opened in enumerate(candidates):
            difference = opened - closed
            if np.count_nonzero(difference == 1) == 1 and np.count_nonzero(difference) == 1:
                pairs.append(
                    AdjacentLayoutPair(
                        closed_layout=closed_index,
                        open_layout=open_index,
                        added_facility=int(np.flatnonzero(difference == 1)[0]),
                    )
                )
    return tuple(pairs)


def marginal_benefit_components(
    closed_recourse_costs: np.ndarray,
    open_recourse_costs: np.ndarray,
    *,
    annual_fixed_cost: float,
    risk_weight: float,
    cvar_alpha: float,
) -> MarginalBenefit:
    """Decompose the net value of adding one facility to a fixed layout."""
    closed = _finite_vector(closed_recourse_costs, "closed_recourse_costs")
    opened = _finite_vector(open_recourse_costs, "open_recourse_costs")
    if closed.shape != opened.shape:
        raise ValueError("paired recourse-cost samples must have the same shape")
    if annual_fixed_cost < 0.0 or not np.isfinite(annual_fixed_cost):
        raise ValueError("annual_fixed_cost must be finite and nonnegative")
    expected_saving = float(np.mean(closed) - np.mean(opened))
    cvar_saving = float(
        empirical_cvar(closed, cvar_alpha) - empirical_cvar(opened, cvar_alpha)
    )
    gross = float(
        (1.0 - risk_weight) * expected_saving + risk_weight * cvar_saving
    )
    return MarginalBenefit(
        expected_recourse_saving=expected_saving,
        cvar_recourse_saving=cvar_saving,
        gross_risk_adjusted_benefit=gross,
        annual_fixed_cost=float(annual_fixed_cost),
        net_opening_benefit=float(gross - annual_fixed_cost),
    )


def crossed_scalar_attribution(values: np.ndarray) -> dict[str, float]:
    """Exact two-way sums-of-squares attribution for a crossed scalar panel."""
    panel = np.asarray(values, float)
    if panel.ndim != 2 or min(panel.shape) < 2 or not np.all(np.isfinite(panel)):
        raise ValueError("values must be a finite (outer>=2, seed>=2) matrix")
    outer_count, seed_count = panel.shape
    grand = float(panel.mean())
    outer_effect = panel.mean(axis=1) - grand
    seed_effect = panel.mean(axis=0) - grand
    interaction = (
        panel
        - grand
        - outer_effect[:, None]
        - seed_effect[None, :]
    )
    outer_ss = float(seed_count * np.sum(outer_effect**2))
    seed_ss = float(outer_count * np.sum(seed_effect**2))
    interaction_ss = float(np.sum(interaction**2))
    total_ss = float(np.sum((panel - grand) ** 2))
    closure = abs(outer_ss + seed_ss + interaction_ss - total_ss)
    if closure > max(1e-8, 1e-12 * max(1.0, total_ss)):
        raise RuntimeError("crossed scalar sums of squares do not close")

    def share(component: float) -> float:
        return component / total_ss if total_ss > 0.0 else 0.0

    return {
        "grand_mean": grand,
        "outer_ss": outer_ss,
        "seed_ss": seed_ss,
        "interaction_ss": interaction_ss,
        "total_ss": total_ss,
        "outer_share": share(outer_ss),
        "seed_share": share(seed_ss),
        "interaction_share": share(interaction_ss),
        "closure_abs": closure,
        "closure_relative": closure / max(1.0, total_ss),
    }


def classify_net_benefit_interval(lower: float, upper: float) -> str:
    """Classify a fixed-cost threshold interval without a significance claim."""
    if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
        raise ValueError("interval endpoints must be finite and ordered")
    if lower > 0.0:
        return "open"
    if upper < 0.0:
        return "close"
    return "ambiguous"


def optimize_empirical_crps_wasserstein_bound(
    forecast_costs: np.ndarray,
    reference_costs: np.ndarray,
    *,
    n_thresholds: int = 257,
) -> EmpiricalTailBound:
    """Optimize the tail-aware CRPS-to-W1 bound for two empirical laws."""
    forecast = _finite_vector(forecast_costs, "forecast_costs")
    reference = _finite_vector(reference_costs, "reference_costs")
    if np.any(forecast < 0.0) or np.any(reference < 0.0) or n_thresholds < 2:
        raise ValueError("costs must be nonnegative and n_thresholds at least two")
    divergence = crps_divergence(forecast, reference)
    combined = np.concatenate([forecast, reference])
    candidates = np.unique(
        np.quantile(combined, np.linspace(0.0, 1.0, n_thresholds))
    )
    candidates = candidates[candidates > 0.0]
    if len(candidates) == 0:
        candidates = np.array([np.finfo(float).tiny])

    best: tuple[float, float, float, float] | None = None
    for threshold in candidates:
        forecast_tail = float(np.mean(np.maximum(forecast - threshold, 0.0)))
        reference_tail = float(np.mean(np.maximum(reference - threshold, 0.0)))
        bound = crps_to_wasserstein_bound(
            divergence,
            float(threshold),
            forecast_tail_excess=forecast_tail,
            reference_tail_excess=reference_tail,
        )
        candidate = (bound, float(threshold), forecast_tail, reference_tail)
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    return EmpiricalTailBound(
        wasserstein_bound=best[0],
        threshold=best[1],
        divergence=divergence,
        forecast_tail_excess=best[2],
        reference_tail_excess=best[3],
    )


def decision_margin_certificate(
    reference_objectives: np.ndarray,
    comparison_objectives: np.ndarray,
) -> DecisionMarginCertificate:
    """Check the sufficient decision-stability condition Delta > 2 epsilon."""
    reference = _finite_vector(reference_objectives, "reference_objectives")
    comparison = _finite_vector(comparison_objectives, "comparison_objectives")
    if reference.shape != comparison.shape or len(reference) < 2:
        raise ValueError("objective vectors must have the same length of at least two")
    order = np.argsort(reference, kind="stable")
    gap = float(reference[order[1]] - reference[order[0]])
    error = float(np.max(np.abs(reference - comparison)))
    reference_best = int(order[0])
    comparison_best = int(np.argmin(comparison))
    return DecisionMarginCertificate(
        reference_best=reference_best,
        comparison_best=comparison_best,
        reference_gap=gap,
        uniform_error=error,
        gap_condition_holds=bool(gap > 2.0 * error),
        optimizer_agrees=bool(reference_best == comparison_best),
    )


def freeze_dciva_candidate_layouts(
    results: Mapping[str, Any], n_facilities: int
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Freeze unique layouts from pre-holdout D-CIVA decision records only."""
    if n_facilities < 1:
        raise ValueError("n_facilities must be positive")
    design = results.get("design", {})
    if int(design.get("train_end", -1)) >= int(design.get("test_years", [0])[0]):
        raise ValueError("D-CIVA design does not separate training and holdout")
    reference = results.get("reference_layout", {})
    sensitivity = results.get("reference_sensitivity", {})
    if reference.get("uses_holdout") is not False:
        raise ValueError("reference layout is not certified pre-holdout")
    if sensitivity.get("uses_holdout_outcomes") is not False:
        raise ValueError("alternative layout is not certified pre-holdout")

    records: list[tuple[str, Any]] = [
        ("reference_layout", reference.get("open_ids", [])),
        ("reference_sensitivity", sensitivity.get("alternative_open_ids", [])),
    ]
    for section in ("crossed_records", "plug_in_records", "mixture_records"):
        records.extend((section, row.get("open_ids", [])) for row in results.get(section, []))

    unique: dict[tuple[int, ...], set[str]] = {}
    for source, open_ids in records:
        signature = tuple(sorted(int(index) for index in open_ids))
        if len(signature) != len(set(signature)) or any(
            index < 0 or index >= n_facilities for index in signature
        ):
            raise ValueError(f"invalid facility identifiers in {source}")
        unique.setdefault(signature, set()).add(source)
    if not unique:
        raise ValueError("no candidate layouts were found")

    signatures = sorted(unique, key=lambda item: (len(item), item))
    layouts = np.zeros((len(signatures), n_facilities), dtype=int)
    labels = []
    for row, signature in enumerate(signatures):
        layouts[row, list(signature)] = 1
        labels.append(
            ",".join(map(str, signature)) + "|" + "+".join(sorted(unique[signature]))
        )
    return layouts, tuple(labels)


def evaluate_recourse_costs(
    problem: FacilityLocationProblem,
    y_open: np.ndarray,
    demands: np.ndarray,
    *,
    backend: str = "scipy",
) -> np.ndarray:
    """Evaluate exact recourse costs for a fixed layout and demand matrix."""
    decision = np.asarray(y_open, float)
    scenarios = np.asarray(demands, float)
    if decision.shape != (problem.n_facilities,) or np.any(
        (decision != 0.0) & (decision != 1.0)
    ):
        raise ValueError("y_open must be a binary facility vector")
    if (
        scenarios.ndim != 2
        or scenarios.shape[1] != problem.n_customers
        or np.any(scenarios < 0.0)
        or not np.all(np.isfinite(scenarios))
    ):
        raise ValueError("demands must be a finite nonnegative scenario matrix")
    if backend == "scipy":
        return _evaluate_recourse_scipy(problem, decision, scenarios)
    if backend == "cplex":
        return _evaluate_recourse_cplex(problem, decision, scenarios)
    raise ValueError(f"unknown backend: {backend}")


def _recourse_matrix_data(
    problem: FacilityLocationProblem, decision: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    open_ids = np.flatnonzero(decision > 0.5)
    n = problem.n_customers
    n_open = len(open_ids)
    transport = (
        problem.cost_per_ton_km * problem.distance_km[:, open_ids]
        + problem.processing_costs[open_ids][None, :]
    )
    objective = np.concatenate(
        [transport.reshape(-1), np.full(n, problem.unmet_penalty)]
    )
    return objective, open_ids, n_open


def _evaluate_recourse_scipy(
    problem: FacilityLocationProblem, decision: np.ndarray, scenarios: np.ndarray
) -> np.ndarray:
    from scipy.optimize import linprog
    from scipy.sparse import lil_matrix

    objective, open_ids, n_open = _recourse_matrix_data(problem, decision)
    n = problem.n_customers
    n_flow = n * n_open
    n_variables = n_flow + n
    equality = lil_matrix((n, n_variables), dtype=float)
    for customer in range(n):
        equality[customer, customer * n_open : (customer + 1) * n_open] = 1.0
        equality[customer, n_flow + customer] = 1.0
    capacity = lil_matrix((n_open, n_variables), dtype=float)
    for facility in range(n_open):
        capacity[facility, facility:n_flow:n_open] = 1.0
    equality = equality.tocsr()
    capacity = capacity.tocsr()
    rhs_capacity = problem.capacities[open_ids]
    costs = np.empty(len(scenarios), float)
    for index, demand in enumerate(scenarios):
        result = linprog(
            objective,
            A_ub=capacity,
            b_ub=rhs_capacity,
            A_eq=equality,
            b_eq=demand,
            bounds=(0.0, None),
            method="highs",
        )
        if not result.success:
            raise RuntimeError(f"recourse LP failed for scenario {index}: {result.message}")
        costs[index] = float(result.fun)
    return costs


def _evaluate_recourse_cplex(
    problem: FacilityLocationProblem, decision: np.ndarray, scenarios: np.ndarray
) -> np.ndarray:
    import cplex

    objective, open_ids, n_open = _recourse_matrix_data(problem, decision)
    n = problem.n_customers
    n_flow = n * n_open
    model = cplex.Cplex()
    model.set_log_stream(None)
    model.set_error_stream(None)
    model.set_warning_stream(None)
    model.set_results_stream(None)
    model.objective.set_sense(model.objective.sense.minimize)
    model.variables.add(
        obj=objective.tolist(),
        lb=[0.0] * len(objective),
    )
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
    for index, demand in enumerate(scenarios):
        if index:
            model.linear_constraints.set_rhs(
                [(customer, float(demand[customer])) for customer in range(n)]
            )
        model.solve()
        if not model.solution.is_primal_feasible():
            raise RuntimeError(f"CPLEX recourse LP failed for scenario {index}")
        costs[index] = float(model.solution.get_objective_value())
    model.end()
    return costs
