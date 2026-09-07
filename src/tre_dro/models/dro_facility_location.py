"""Problem data for the two-stage facility-location formulation.

Problem data is separate from ambiguity-set calibration. Exact solution of the
full generalized-moment recourse problem is outside this data container.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data.schema import Instance


@dataclass(frozen=True)
class FacilityLocationProblem:
    """Deterministic problem data for a capacitated facility-location instance."""

    fixed_costs: np.ndarray      # (m,) facility opening costs
    capacities: np.ndarray       # (m,) facility capacities
    processing_costs: np.ndarray # (m,) facility processing costs (yuan/ton)
    cost_per_ton_km: float       # yuan/(ton·km)
    distance_km: np.ndarray      # (n, m) haversine distance matrix
    max_open: int                # cardinality constraint (∑ x_j ≤ K)
    unmet_penalty: float         # yuan/ton for unmet demand (lost-sale)

    @classmethod
    def from_instance(cls, inst: Instance) -> "FacilityLocationProblem":
        return cls(
            fixed_costs=inst.fixed_costs,
            capacities=inst.capacities,
            processing_costs=inst.processing_costs,
            cost_per_ton_km=inst.cost_per_ton_km,
            distance_km=inst.distance_km,
            max_open=inst.max_open,
            unmet_penalty=inst.unmet_penalty,
        )

    @property
    def n_customers(self) -> int:
        return self.distance_km.shape[0]

    @property
    def n_facilities(self) -> int:
        return self.distance_km.shape[1]
