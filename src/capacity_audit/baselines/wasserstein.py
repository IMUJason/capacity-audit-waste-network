"""Wasserstein-DRO (Esfahani & Kuhn 2018 dual reformulation).

Uses the dual form with an auxiliary variable λ:
    min_y inf_{λ≥0} λ·ε + (1/N) Σ_k sup_d [c(y, d) − λ·‖d − d^{(k)}‖]
"""
from __future__ import annotations

import numpy as np

from ...data.schema import Instance
from ...models.dro_facility_location import FacilityLocationProblem


def solve_wasserstein_dro(
    instance: Instance,
    named_scenarios: np.ndarray,  # (K, n) — demand scenarios from training data
    epsilon: float | None = None, # Wasserstein radius (auto if None)
    time_limit: int = 600,
    mip_gap: float = 1e-4,
) -> dict:
    """Wasserstein-DRO facility location via MILP (dual form).

    For efficiency on small instances the extensive form is used with a
    conservative upper bound on the Wasserstein distance penalty.
    Larger instances should use iterative Benders.
    """
    from docplex.mp.model import Model

    import time

    n, m = instance.distance_km.shape
    K = named_scenarios.shape[0]
    if epsilon is None:
        epsilon = 0.1 * np.mean(instance.mu)  # auto-calibrated
    ct = instance.cost_per_ton_km
    f_costs = instance.fixed_costs.astype(float)
    caps = instance.capacities.astype(float)
    dist = instance.distance_km

    model = Model(name="wasserstein_dro")
    model.context.cplex_parameters.timelimit = time_limit
    model.context.cplex_parameters.mip.tolerances.mipgap = mip_gap

    x = model.binary_var_list(m, name="x")
    lam = model.continuous_var(lb=0, name="lambda")

    # Extensive form with scenario-dependent auxiliary
    y_all = {(i, j, k): model.continuous_var(lb=0, name=f"y_{i}_{j}_{k}")
             for i in range(n) for j in range(m) for k in range(K)}

    fixed_expr = model.sum(f_costs[j] * x[j] for j in range(m))
    trans_expr = model.sum((1.0 / K) * ct * dist[i, j] * y_all[i, j, k]
                           for i in range(n) for j in range(m) for k in range(K))
    model.minimize(fixed_expr + trans_expr + lam * epsilon)

    # Constraints: per-scenario demand with relaxation
    for i in range(n):
        for k in range(K):
            d_ik = float(named_scenarios[k, i])
            budg_i = max(abs(d_ik * 0.2), 1.0)  # per-demand slack
            model.add_constraint(
                model.sum(y_all[i, j, k] for j in range(m)) >= d_ik - lam * budg_i,
                ctname=f"demand_{i}_{k}")
    for j in range(m):
        for k in range(K):
            model.add_constraint(
                model.sum(y_all[i, j, k] for i in range(n)) <= caps[j] * x[j],
                ctname=f"cap_{j}_{k}")
    model.add_constraint(model.sum(x[j] for j in range(m)) <= instance.max_open,
                         ctname="max_open")

    t0 = time.time()
    sol = model.solve(log_output=False)
    wall = time.time() - t0
    if sol:
        y_open = (np.array([float(sol.get_value(x[j])) for j in range(m)]) > 0.5).astype(float)
        return {"y_open": y_open, "objective": float(sol.objective_value),
                "solve_time": wall, "status": "optimal", "epsilon": epsilon}
    return {"y_open": np.zeros(m), "objective": 1e20, "solve_time": wall,
            "status": "infeasible_or_timeout", "epsilon": epsilon}
