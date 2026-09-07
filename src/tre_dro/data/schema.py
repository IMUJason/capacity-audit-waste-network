"""Plain dataclasses for tre_dro (no pydantic dependency; keeps numpy arrays)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Moments:
    """Location-specific first/second-moment estimates."""

    mu: np.ndarray       # (n,) location means
    sigma: np.ndarray    # (n,) location standard deviations

    @property
    def cv(self) -> np.ndarray:
        return self.sigma / np.maximum(self.mu, 1e-9)

    @property
    def n(self) -> int:
        return int(self.mu.shape[0])


@dataclass(frozen=True)
class Customer:
    id: int
    lon: float
    lat: float
    mu: float
    sigma: float
    name: str = ""


@dataclass(frozen=True)
class Facility:
    id: int
    lon: float
    lat: float
    fixed_cost: float
    capacity: float
    name: str = ""
    processing_cost: float = 0.0  # per-ton processing cost (yuan/ton)


@dataclass
class Instance:
    """A facility-location instance (deterministic input)."""

    name: str
    customers: tuple[Customer, ...]
    facilities: tuple[Facility, ...]
    distance_km: np.ndarray          # (n_customers, n_facilities)
    cost_per_ton_km: float           # yuan/(ton*km)
    max_open: int
    unmet_penalty: float             # yuan/ton for unmet demand
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def n_customers(self) -> int:
        return len(self.customers)

    @property
    def n_facilities(self) -> int:
        return len(self.facilities)

    @property
    def mu(self) -> np.ndarray:
        return np.array([c.mu for c in self.customers])

    @property
    def sigma(self) -> np.ndarray:
        return np.array([c.sigma for c in self.customers])

    @property
    def fixed_costs(self) -> np.ndarray:
        return np.array([f.fixed_cost for f in self.facilities])

    @property
    def capacities(self) -> np.ndarray:
        return np.array([f.capacity for f in self.facilities])

    @property
    def processing_costs(self) -> np.ndarray:
        return np.array([f.processing_cost for f in self.facilities])


@dataclass
class Solution:
    """First-stage solution of a method on an instance."""

    method: str
    y_open: np.ndarray               # (m,) binary facility-open decisions
    objective: float                 # method's own objective (definition varies)
    solve_time: float
    status: str = "optimal"
    iterations: int | None = None    # Benders iterations, else None
    gap: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)
