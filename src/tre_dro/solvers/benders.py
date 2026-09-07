"""Dual-linearized lower-bound diagnostic for moment-robust recourse.

Cleaned from ``code/_archive_unused_scripts/benders_dro.py``. The critical
repair over the old code:

- **DELETED** ``benders_dro.py:377-386`` where the DRO subproblem *replaced
  demand with μ+Ωσ* (pseudo-DRO). The subproblem now solves the **nominal**
  recourse LP and returns dual variables π.
- **RE-ACTIVATED** ``robust_cuts.py:165-180`` ``compute_robust_norm``, now
  generalised to full spatial covariance Σ (not just diagonal σ²).
- **DUAL BOUND**  theta >= alpha' mu + kappa ||Sigma^(1/2) alpha||
  + lambda' M y, with signed capacity duals lambda <= 0.

For a fixed dual vector the norm is a scalar constant, so the master is a MILP.
The bound is valid for each affine recourse piece, but nominal dual generation
is not an exact oracle for the expectation of piecewise-linear recourse. This
class therefore reports a diagnostic lower bound and never an optimal DRO gap.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..ambiguity.ambiguity_set import AmbiguitySet
from ..models.dro_facility_location import FacilityLocationProblem
from ..models.recourse import solve_nominal_recourse, RecourseSolution


@dataclass
class BendersConfig:
    max_iter: int = 200
    tolerance: float = 1e-4
    time_limit: int = 1800  # seconds per solve
    mip_gap: float = 1e-4
    cut_strategy: str = "most_violated"


@dataclass
class BendersSolution:
    y_open: np.ndarray
    objective: float
    lower_bound: float
    upper_bound: float
    iterations: int
    solve_time: float
    status: str
    gap: float
    cuts: int
    history: list[dict] = field(default_factory=list)


class BendersSolver:
    """Generate covariance-aware dual lower bounds for facility decisions.

    Master (MILP):    min_y  Σ f_j y_j + θ
                       s.t.  Σ y_j ≤ K, θ ≥ … (Benders cuts)
    Subproblem:  nominal recourse LP for fixed y → duals (α, β).
    Dual bound: theta >= alpha' mu + kappa ||Sigma^(1/2) alpha||
                + lambda' M y.
    """

    def __init__(
        self,
        problem: FacilityLocationProblem,
        amb: AmbiguitySet,
        config: BendersConfig | None = None,
    ):
        self.problem = problem
        self.amb = amb
        self.config = config or BendersConfig()

    def solve(self, warmstart: np.ndarray | None = None) -> BendersSolution:
        import time

        t0 = time.time()
        n, m = self.problem.distance_km.shape
        f_costs = np.asarray(self.problem.fixed_costs, float)
        max_open = self.problem.max_open

        from docplex.mp.model import Model

        master = Model(name="benders_master")
        master.context.cplex_parameters.timelimit = self.config.time_limit
        master.context.cplex_parameters.mip.tolerances.mipgap = self.config.mip_gap

        y = master.binary_var_list(m, name="y")
        theta = master.continuous_var(lb=0, name="theta")
        master.add_constraint(master.sum(y[j] for j in range(m)) <= max_open, ctname="card")
        master.minimize(master.sum(f_costs[j] * y[j] for j in range(m)) + theta)

        lb = -np.inf
        best_y = np.zeros(m)
        history: list[dict] = []
        cuts_count = 0
        status = "failed"

        if warmstart is not None:
            master.add_mip_start(
                master.new_solution({y[j]: float(warmstart[j]) for j in range(m)}))
        visited: set[tuple[int, ...]] = set()

        for iteration in range(1, self.config.max_iter + 1):
            # --- Master step ---
            m_sol = master.solve(log_output=False)
            if not m_sol:
                break
            y_current = np.array([float(m_sol.get_value(y[j])) for j in range(m)])
            theta_val = float(m_sol.get_value(theta))
            lb = float(m_sol.objective_value)  # lower bound
            y_int = (y_current > 0.5).astype(float)
            # Round fractional: small instances may have integer anyway
            y_int_round = y_int.copy()
            y_key = tuple(y_int_round.astype(int).tolist())

            # --- Subproblem: nominal recourse LP for y_int ---
            rec_nom = solve_nominal_recourse(
                self.problem, y_int_round,
                demand=self.amb.mu,  # nominal demand = μ (not worst-case!)
                backend="scipy",
            )

            # Nominal cost at current y
            nominal_cost = rec_nom.total_cost

            # --- Robust term ---
            from ..ambiguity.covariance import robust_norm
            pi_d = rec_nom.pi_demand
            if not np.all(np.isfinite(pi_d)):
                raise RuntimeError("recourse solver did not return finite demand duals")
            aux_vec = pi_d
            rn = robust_norm(self.amb.sigma_half, aux_vec)
            robust_term = self.amb.omega * rn

            pi_cap = rec_nom.pi_capacity
            score = float(np.sum(f_costs * y_int_round)) + nominal_cost + robust_term
            best_y = y_int_round.copy()
            history.append({
                "iter": iteration,
                "master_lower_bound": lb,
                "dual_linearized_score": score,
                "y": list(y_key),
            })

            if y_key in visited:
                status = "diagnostic_converged"
                break
            visited.add(y_key)

            # --- Generate robust Benders cut ---
            # θ ≥ Σ α_i μ_i + Ω‖Σ^{1/2}α‖ − Σ_j β_j M_j y_j
            cut_constant = float(np.sum(pi_d * self.amb.mu)) + robust_term
            capacity_expr = master.sum(
                float(pi_cap[j] * self.problem.capacities[j]) * y[j]
                for j in range(m)
            )
            master.add_constraint(
                theta >= cut_constant + capacity_expr,
                ctname=f"dual_bound_{iteration}",
            )
            cuts_count += 1
        else:
            status = "max_iter"

        wall = time.time() - t0
        if not np.isfinite(lb):
            status = "failed"
        return BendersSolution(
            y_open=best_y,
            objective=float(lb),
            lower_bound=float(lb),
            upper_bound=float("nan"),
            iterations=iteration,
            solve_time=wall,
            status=status,
            gap=float("nan"),
            cuts=cuts_count,
            history=history,
        )
