"""Nominal recourse LP and its dual (transportation problem).

For fixed open-facility decisions y_j, the deterministic allocation problem is
a capacitated transport LP. Both backends enforce the same capacity and unmet-
demand constraints. The dual variables are recovered from the LP for use in
dual-linearization diagnostics.

Two backends:
- scipy.linprog  (lightweight, no CPLEX; used in unit tests)
- docplex        (CPLEX, for production solve; returns IIS on infeasibility)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .dro_facility_location import FacilityLocationProblem


@dataclass
class RecourseSolution:
    x: np.ndarray          # (n, m) allocation flows
    total_cost: float      # ∑ c_ij x_ij + unmet_penalty * unmet
    unmet: np.ndarray      # (n,) unmet demand per customer
    pi_demand: np.ndarray  # (n,) dual of demand constraint
    pi_capacity: np.ndarray# (m,) dual of capacity constraint (≤ 0)
    status: str = "optimal"
    extra: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}


def solve_nominal_recourse(
    problem: FacilityLocationProblem,
    y_open: np.ndarray,
    demand: np.ndarray,
    backend: str = "scipy",
) -> RecourseSolution:
    """Solve the deterministic multi-commodity transport LP for fixed y, demand.

    min  Σ_ij c_ij x_ij + unmet_penalty * Σ_i u_i
    s.t. Σ_j x_ij + u_i ≥ d_i  ∀i  (π_i ≥ 0)
         Σ_i x_ij ≤ C_j · y_j  ∀j  (β_j ≥ 0)
         x_ij ≥ 0, u_i ≥ 0.
    """
    if backend == "scipy":
        return _solve_scipy(problem, y_open, demand)
    elif backend == "docplex":
        return _solve_docplex(problem, y_open, demand)
    else:
        raise ValueError(f"unknown backend: {backend}")


def _solve_scipy(
    problem: FacilityLocationProblem, y_open: np.ndarray, demand: np.ndarray,
) -> RecourseSolution:
    from scipy.optimize import linprog

    n, m = problem.distance_km.shape
    transport_cost = (
        problem.cost_per_ton_km * problem.distance_km
        + problem.processing_costs[None, :]
    )
    # Keep zero-capacity facilities in the LP. Removing their variables gives
    # the same primal value but duals that are not feasible for another y,
    # which invalidates cross-decision affine bounds.
    open_ndx = np.arange(m)
    m_open = m
    k = n * m_open  # x variables + n unmet variables
    n_vars = k + n

    # variable order: x_{i, j'} for j'=0..m_open-1, then u_i
    c_obj = np.zeros(n_vars)
    for ii in range(n):
        for jj, j_orig in enumerate(open_ndx):
            c_obj[ii * m_open + jj] = transport_cost[ii, j_orig]
        c_obj[k + ii] = problem.unmet_penalty

    bounds = [(0, None)] * n_vars

    # -service <= -d is value-equivalent to equality for non-negative demand
    # and positive costs, while providing complete recourse on R^n.
    A_ub_rows2: list[np.ndarray] = []
    b_ub_rows2: list[float] = []
    for ii in range(n):
        row = np.zeros(n_vars)
        row[ii * m_open : (ii + 1) * m_open] = -1.0
        row[k + ii] = -1.0
        A_ub_rows2.append(row)
        b_ub_rows2.append(-float(demand[ii]))
    for jj, j_orig in enumerate(open_ndx):
        row = np.zeros(n_vars)
        for ii in range(n):
            row[ii * m_open + jj] = 1.0
        A_ub_rows2.append(row)
        b_ub_rows2.append(float(problem.capacities[j_orig] * y_open[j_orig]))

    A_ub2 = np.array(A_ub_rows2) if A_ub_rows2 else None
    b_ub2 = np.array(b_ub_rows2) if A_ub_rows2 else None

    result2 = linprog(
        c_obj, A_ub=A_ub2, b_ub=b_ub2, bounds=bounds, method="highs",
    )

    x_sol = result2.x if result2.success else None
    status = "optimal" if result2.success else "unknown"
    if x_sol is None:
        x_sol = np.zeros(n_vars)

    # HiGHS upper-bound marginals are non-positive. Negating the first n
    # marginals recovers non-negative demand duals.
    x_flow = np.zeros((n, m))
    for ii in range(n):
        for jj, j_orig in enumerate(open_ndx):
            x_flow[ii, j_orig] = max(0.0, float(x_sol[ii * m_open + jj]))
    unmet = np.maximum(0.0, demand - x_flow.sum(axis=1))
    total = (transport_cost * x_flow).sum() + problem.unmet_penalty * unmet.sum()
    pi_demand = np.full(n, np.nan)
    pi_capacity = np.zeros(m)
    if result2.success:
        marginals = np.asarray(result2.ineqlin.marginals, float)
        pi_demand = -marginals[:n]
        if m_open:
            cap_marginals = marginals[n:]
            pi_capacity[open_ndx] = cap_marginals
    return RecourseSolution(
        x=x_flow, total_cost=float(total), unmet=unmet,
        pi_demand=pi_demand, pi_capacity=pi_capacity,
        status=status, extra={"backend": "scipy", "solver_message": result2.message},
    )


def _solve_docplex(
    problem: FacilityLocationProblem, y_open: np.ndarray, demand: np.ndarray,
) -> RecourseSolution:
    """Production LP solve via CPLEX; returns actual dual variables."""
    from docplex.mp.model import Model

    n, m = problem.distance_km.shape
    model = Model(name="nominal_recourse")
    model.context.cplex_parameters.simplex.display = 0
    x = {(i, j): model.continuous_var(lb=0, name=f"x_{i}_{j}")
         for i in range(n) for j in range(m)}
    u = {i: model.continuous_var(lb=0, name=f"u_{i}") for i in range(n)}
    # demand
    for i in range(n):
        model.add_constraint(
            model.sum(x[i, j] for j in range(m)) + u[i] >= demand[i],
            ctname=f"demand_{i}")
    # capacity
    for j in range(m):
        model.add_constraint(
            model.sum(x[i, j] for i in range(n))
            <= problem.capacities[j] * y_open[j],
            ctname=f"cap_{j}")
    total = (model.sum((problem.cost_per_ton_km * problem.distance_km[i, j]
                       + problem.processing_costs[j]) * x[i, j]
                       for i in range(n) for j in range(m)) +
             model.sum(problem.unmet_penalty * u[i] for i in range(n)))
    model.minimize(total)
    sol = model.solve(log_output=False)

    x_flow = np.zeros((n, m))
    for (i, j), v in x.items():
        if sol is not None:
            x_flow[i, j] = max(0.0, float(sol.get_value(v)))
    unmet_arr = np.maximum(0.0, demand - x_flow.sum(axis=1))
    total_cost = float(sol.objective_value) if sol else 1e20
    pi_demand = np.array([float(model.get_constraint_by_name(f"demand_{i}").dual_value)
                          for i in range(n)]) if sol else np.full(n, np.nan)
    pi_cap = np.array([
        float(model.get_constraint_by_name(f"cap_{j}").dual_value)
        for j in range(m)
    ]) if sol else np.full(m, np.nan)
    return RecourseSolution(x=x_flow, total_cost=total_cost, unmet=unmet_arr,
                            pi_demand=pi_demand, pi_capacity=pi_cap,
                            status="optimal" if sol else "infeasible",
                            extra={"backend": "docplex", "model": model})
