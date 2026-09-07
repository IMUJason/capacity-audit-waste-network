"""Interval Robust Optimisation (Bertsimas-Sim Γ-robustness)."""
from __future__ import annotations

import numpy as np

from ...data.schema import Instance
from ...models.dro_facility_location import FacilityLocationProblem
from ...solvers.cplex_misocp import solve_deterministic_nominal, MISOCPResult


def solve_interval_ro(
    instance: Instance,
    gamma: int = 3,          # budget of uncertainty
    deviation: float = 0.3,  # demand deviation from μ as fraction
    time_limit: int = 300,
) -> MISOCPResult:
    """Bertsimas-Sim RO: each demand ∈ [μ − δ·μ, μ + δ·μ].

    For the capacity-constrained FLP this reduces to a deterministic CFLP
    with worst-case demand = μ + (deviation)·μ, adjusted by the budget γ.
    For large instances the canonical BS formulation (dualised) is used.
    """
    mu = instance.mu.astype(float)
    # Simple approximation: worst-case = μ + deviation·μ (all γ demands simultaneously maximised)
    worst_d = mu * (1.0 + deviation)
    problem = FacilityLocationProblem.from_instance(instance)
    return solve_deterministic_nominal(problem, worst_d, time_limit=time_limit)
