"""Direct deterministic capacitated facility-location solver via CPLEX.

This module does not implement a moment-DRO MISOCP. It solves a deterministic
instance for an explicitly supplied demand vector and is used for the nominal
baseline and clearly labelled stress-demand proxies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..ambiguity.ambiguity_set import AmbiguitySet
from ..data.schema import Instance
from ..models.dro_facility_location import FacilityLocationProblem


@dataclass
class DeterministicResult:
    y_open: np.ndarray       # (m,) binary
    objective: float
    solve_time: float
    status: str
    gap: float | None = None
    extra: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}


def solve_deterministic_nominal(
    problem: FacilityLocationProblem,
    demand: np.ndarray,
    time_limit: int = 300,
    mip_gap: float = 1e-4,
) -> DeterministicResult:
    """Solve deterministic CFLP with capacity-constrained unmet-demand recourse."""
    from docplex.mp.model import Model

    n, m = problem.distance_km.shape
    ct = problem.cost_per_ton_km
    d_arr = np.asarray(demand, float)
    f_costs = np.asarray(problem.fixed_costs, float)
    caps = np.asarray(problem.capacities, float)
    dist = problem.distance_km

    model = Model(name="deterministic_cflp")
    model.context.cplex_parameters.timelimit = time_limit
    model.context.cplex_parameters.mip.tolerances.mipgap = mip_gap

    x = model.binary_var_list(m, name="x")
    y = {(i, j): model.continuous_var(lb=0, name=f"y_{i}_{j}")
         for i in range(n) for j in range(m)}
    u = model.continuous_var_list(n, lb=0, name="u")

    # Objective
    fixed_expr = model.sum(f_costs[j] * x[j] for j in range(m))
    trans_expr = model.sum(
        (ct * dist[i, j] + problem.processing_costs[j]) * y[i, j]
        for i in range(n) for j in range(m)
    )
    unmet_expr = model.sum(problem.unmet_penalty * u[i] for i in range(n))
    model.minimize(fixed_expr + trans_expr + unmet_expr)

    # Demand
    for i in range(n):
        model.add_constraint(
            model.sum(y[i, j] for j in range(m)) + u[i] == d_arr[i],
            ctname=f"demand_{i}")
    # Capacity
    for j in range(m):
        model.add_constraint(
            model.sum(y[i, j] for i in range(n)) <= caps[j] * x[j],
            ctname=f"cap_{j}")
    # Cardinality
    model.add_constraint(model.sum(x[j] for j in range(m)) <= problem.max_open,
                         ctname="max_open")

    import time
    t0 = time.time()
    sol = model.solve(log_output=False)
    wall = time.time() - t0

    if sol:
        y_open = np.array([float(sol.get_value(x[j])) for j in range(m)])
        return DeterministicResult(
            y_open=(y_open > 0.5).astype(float),
            objective=float(sol.objective_value),
            solve_time=wall,
            status="optimal",
            gap=float(sol.mip_gap) if hasattr(sol, 'mip_gap') else None,
        )
    return DeterministicResult(
        y_open=np.zeros(m), objective=1e20, solve_time=wall,
        status="infeasible_or_timeout",
    )


# Backward-compatible alias for archived scripts. New code should use the
# truthful DeterministicResult name.
MISOCPResult = DeterministicResult
