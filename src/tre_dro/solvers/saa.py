"""True two-stage Sample Average Approximation (SAA) solver.

This replaces the old pseudo-SAA (which just used σ=0 and solved a
deterministic CFLP once). Genuine SAA:
    min_y  fᵀy + (1/N) Σ_{k=1}^N  Q(y, d^{(k)})
where d^{(k)} are i.i.d. demand scenarios.

Q(y, d) is the transport LP (nominal recourse).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..data.schema import Instance
from ..models.dro_facility_location import FacilityLocationProblem
from ..models.recourse import solve_nominal_recourse


@dataclass
class SAAResult:
    y_open: np.ndarray
    objective: float
    solve_time: float
    status: str
    scenarios: int
    extra: dict[str, Any] = field(default_factory=dict)


def solve_saa(
    problem: FacilityLocationProblem,
    scenarios: np.ndarray,  # shape (n_scenarios, n_customers)
    time_limit: int = 600,
    mip_gap: float = 1e-4,
) -> SAAResult:
    """Solve the two-stage SAA problem via the extensive form MILP.

    Builds a large MILP: for each scenario k, replicate y_ij(k).
    For small instances this is exact; for large instances use Benders/L-shaped.
    """
    from docplex.mp.model import Model

    import time

    n, m = problem.distance_km.shape
    K = scenarios.shape[0]
    ct = problem.cost_per_ton_km
    f_costs = np.asarray(problem.fixed_costs, float)
    caps = np.asarray(problem.capacities, float)
    dist = problem.distance_km

    model = Model(name="saa_extensive")
    model.context.cplex_parameters.timelimit = time_limit
    model.context.cplex_parameters.mip.tolerances.mipgap = mip_gap

    x = model.binary_var_list(m, name="x")
    y_all = {(i, j, k): model.continuous_var(lb=0, name=f"y_{i}_{j}_{k}")
             for i in range(n) for j in range(m) for k in range(K)}
    u_all = {(i, k): model.continuous_var(lb=0, name=f"u_{i}_{k}")
             for i in range(n) for k in range(K)}

    fixed_expr = model.sum(f_costs[j] * x[j] for j in range(m))
    trans_expr = model.sum(
        (1.0 / K) * (ct * dist[i, j] + problem.processing_costs[j]) * y_all[i, j, k]
        for i in range(n) for j in range(m) for k in range(K))
    unmet_expr = model.sum(
        (1.0 / K) * problem.unmet_penalty * u_all[i, k]
        for i in range(n) for k in range(K))
    model.minimize(fixed_expr + trans_expr + unmet_expr)

    for i in range(n):
        for k in range(K):
            d_ik = float(scenarios[k, i])
            model.add_constraint(
                model.sum(y_all[i, j, k] for j in range(m)) + u_all[i, k] == d_ik,
                ctname=f"demand_{i}_{k}")
    for j in range(m):
        for k in range(K):
            model.add_constraint(
                model.sum(y_all[i, j, k] for i in range(n)) <= caps[j] * x[j],
                ctname=f"cap_{j}_{k}")
    model.add_constraint(model.sum(x[j] for j in range(m)) <= problem.max_open,
                         ctname="max_open")

    t0 = time.time()
    sol = model.solve(log_output=False)
    wall = time.time() - t0

    if sol:
        y_open = (np.array([float(sol.get_value(x[j])) for j in range(m)]) > 0.5).astype(float)
        return SAAResult(
            y_open=y_open, objective=float(sol.objective_value),
            solve_time=wall, status="optimal", scenarios=K,
        )
    return SAAResult(y_open=np.zeros(m), objective=1e20, solve_time=wall,
                     status="infeasible_or_timeout", scenarios=K)
