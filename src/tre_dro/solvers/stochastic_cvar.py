"""Exact extensive-form mean-CVaR two-stage facility-location MILP."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..models.dro_facility_location import FacilityLocationProblem


@dataclass
class MeanCVaRResult:
    y_open: np.ndarray
    objective: float
    expected_recourse: float
    cvar_recourse: float
    solve_time: float
    status: str
    mip_gap: float | None
    scenarios: int
    extra: dict[str, Any] = field(default_factory=dict)


def solve_mean_cvar_saa(
    problem: FacilityLocationProblem,
    scenarios: np.ndarray,
    *,
    risk_weight: float = 0.5,
    cvar_alpha: float = 0.9,
    fixed_cost_multiplier: float = 1.0,
    time_limit: int = 900,
    mip_gap: float = 1e-6,
) -> MeanCVaRResult:
    """Solve the equal-probability extensive form to CPLEX optimality."""
    from docplex.mp.model import Model
    import time

    demand = np.asarray(scenarios, float)
    if demand.ndim != 2 or demand.shape[1] != problem.n_customers:
        raise ValueError("scenarios must have shape (S, n_customers)")
    if not 0.0 <= risk_weight <= 1.0 or not 0.0 < cvar_alpha < 1.0:
        raise ValueError("invalid risk parameters")
    s_count, n = demand.shape
    m = problem.n_facilities
    unit_cost = (
        problem.cost_per_ton_km * problem.distance_km
        + problem.processing_costs[None, :]
    )

    model = Model(name="mean_cvar_facility_location")
    model.context.cplex_parameters.timelimit = time_limit
    model.context.cplex_parameters.mip.tolerances.mipgap = mip_gap
    open_var = model.binary_var_list(m, name="open")
    flow = {
        (s, i, j): model.continuous_var(lb=0, name=f"flow_{s}_{i}_{j}")
        for s in range(s_count) for i in range(n) for j in range(m)
    }
    unmet = {
        (s, i): model.continuous_var(lb=0, name=f"unmet_{s}_{i}")
        for s in range(s_count) for i in range(n)
    }

    recourse = []
    for s in range(s_count):
        recourse.append(
            model.sum(unit_cost[i, j] * flow[s, i, j] for i in range(n) for j in range(m))
            + model.sum(problem.unmet_penalty * unmet[s, i] for i in range(n))
        )
        for i in range(n):
            model.add_constraint(
                model.sum(flow[s, i, j] for j in range(m)) + unmet[s, i]
                == float(demand[s, i])
            )
        for j in range(m):
            model.add_constraint(
                model.sum(flow[s, i, j] for i in range(n))
                <= float(problem.capacities[j]) * open_var[j]
            )

    model.add_constraint(model.sum(open_var) <= problem.max_open)
    expected = model.sum(recourse) / s_count
    eta = model.continuous_var(lb=0, name="cvar_eta")
    excess = model.continuous_var_list(s_count, lb=0, name="cvar_excess")
    for s in range(s_count):
        model.add_constraint(excess[s] >= recourse[s] - eta)
    cvar = eta + model.sum(excess) / ((1.0 - cvar_alpha) * s_count)
    fixed = model.sum(
        fixed_cost_multiplier * float(problem.fixed_costs[j]) * open_var[j]
        for j in range(m)
    )
    model.minimize(fixed + (1.0 - risk_weight) * expected + risk_weight * cvar)

    start = time.time()
    solution = model.solve(log_output=False)
    wall = time.time() - start
    if solution is None:
        return MeanCVaRResult(
            y_open=np.zeros(m), objective=float("inf"), expected_recourse=float("nan"),
            cvar_recourse=float("nan"), solve_time=wall, status="infeasible_or_timeout",
            mip_gap=None, scenarios=s_count,
        )

    opened = np.array([solution.get_value(var) for var in open_var]) > 0.5
    scenario_costs = np.array([solution.get_value(expr) for expr in recourse], float)
    eta_value = float(solution.get_value(eta))
    excess_values = np.array([solution.get_value(var) for var in excess], float)
    empirical_cvar = float(
        eta_value + excess_values.sum() / ((1.0 - cvar_alpha) * s_count)
    )
    solve_details = solution.solve_details
    relative_gap = float(solve_details.mip_relative_gap)
    hit_limit = bool(solve_details.has_hit_limit())
    status = "optimal" if not hit_limit and relative_gap <= mip_gap + 1e-12 else "feasible"
    return MeanCVaRResult(
        y_open=opened.astype(float),
        objective=float(solution.objective_value),
        expected_recourse=float(np.mean(scenario_costs)),
        cvar_recourse=empirical_cvar,
        solve_time=wall,
        status=status,
        mip_gap=relative_gap,
        scenarios=s_count,
        extra={
            "scenario_recourse_costs": scenario_costs.tolist(),
            "cvar_eta": eta_value,
            "cvar_excess": excess_values.tolist(),
            "solver_status": solve_details.status,
            "hit_limit": hit_limit,
        },
    )
