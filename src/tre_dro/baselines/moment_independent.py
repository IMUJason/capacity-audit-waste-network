"""Moment-DRO homogeneous and independent baseline wrappers.

These are thin wrappers around the deterministic CFLP solver, using a specific
AmbiguitySet kind. The 'spatial-DRO' claims of the old code are mapped to
``moment_independent``.
"""
from __future__ import annotations

from ...data.schema import Instance
from ...ambiguity.ambiguity_set import AmbiguitySet
from ...models.dro_facility_location import FacilityLocationProblem
from ...solvers.cplex_misocp import solve_deterministic_nominal, MISOCPResult


def solve_homogeneous_dro(
    instance: Instance, amb: AmbiguitySet, time_limit: int = 300,
) -> MISOCPResult:
    """DRO with Σ = diag((CV̄·μ)²) — the 'old Homo-DRO'."""
    # Robust demand: μ + Ω·σ̄ where σ̄ = μ·CV̄
    robust_d = amb.mu + amb.omega * np.sqrt(np.diag(amb.sigma_matrix))
    return solve_deterministic_nominal(
        FacilityLocationProblem.from_instance(instance), robust_d,
        time_limit=time_limit)

def solve_independent_dro(
    instance: Instance, amb: AmbiguitySet, time_limit: int = 300,
) -> MISOCPResult:
    """DRO with Σ = diag(σ²) — per-location σ, no correlation."""
    robust_d = amb.mu + amb.omega * np.sqrt(np.diag(amb.sigma_matrix))
    return solve_deterministic_nominal(
        FacilityLocationProblem.from_instance(instance), robust_d,
        time_limit=time_limit)


import numpy as np
