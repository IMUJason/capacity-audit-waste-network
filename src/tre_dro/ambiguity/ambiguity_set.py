"""Unified ``AmbiguitySet``: the single object every method consumes.

Three kinds share the same Delage-Ye (2010) form and differ only in Sigma:
- ``homogeneous``   -- Sigma = diag((CV*mu)^2); one global CV.
- ``independent``   -- Sigma = diag(sigma^2); per-location variance.
- ``joint_spatial`` -- Sigma = sigma_i sigma_j exp(-h_ij/rho).

All methods see identical mu_i, sigma_i; only the covariance structure
differs, which isolates the covariance effect on decisions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ..data.schema import Moments
from .covariance import (
    assert_psd,
    build_spatial_sigma,
    cholesky_sqrt,
    diagonal_sigma,
    homogeneous_sigma,
)
from .omega import omega_from_gamma

Kind = Literal["homogeneous", "independent", "joint_spatial"]


@dataclass(frozen=True)
class AmbiguitySet:
    kind: Kind
    mu: np.ndarray               # (n,) location means
    sigma_matrix: np.ndarray     # (n,n) covariance
    gamma1: float                # mean-deviation ellipsoid radius
    gamma2: float                # covariance-inflation factor
    rho_km: float | None = None  # spatial range (joint_spatial only)
    support: tuple[np.ndarray, np.ndarray] | None = None  # (lb, ub) if bounded

    @property
    def n(self) -> int:
        return int(self.mu.shape[0])

    @property
    def omega(self) -> float:
        """Worst-case linear-loss coefficient sqrt(min(gamma1, gamma2))."""
        return omega_from_gamma(self.gamma1, self.gamma2)

    @property
    def sigma_half(self) -> np.ndarray:
        """Σ^{1/2} (PSD-safe)."""
        return cholesky_sqrt(self.sigma_matrix)

    def describe(self) -> dict[str, Any]:
        from .covariance import condition_number, min_eigval

        return {
            "kind": self.kind,
            "n": self.n,
            "gamma1": self.gamma1,
            "gamma2": self.gamma2,
            "omega": self.omega,
            "rho_km": self.rho_km,
            "sigma_min_eig": min_eigval(self.sigma_matrix),
            "sigma_cond": condition_number(self.sigma_matrix),
        }

    @classmethod
    def from_moments(
        cls,
        moments: Moments,
        kind: Kind,
        gamma1: float,
        gamma2: float,
        haversine_km: np.ndarray | None = None,
        rho_km: float | None = None,
        kernel: str = "exponential",
    ) -> "AmbiguitySet":
        mu = np.asarray(moments.mu, float)
        if kind == "homogeneous":
            Sigma = homogeneous_sigma(moments)
        elif kind == "independent":
            Sigma = diagonal_sigma(moments)
        elif kind == "joint_spatial":
            if haversine_km is None or rho_km is None:
                raise ValueError("joint_spatial requires haversine_km and rho_km")
            Sigma = build_spatial_sigma(moments, haversine_km, rho_km, kernel=kernel)
        else:
            raise ValueError(f"unknown kind: {kind}")
        assert_psd(Sigma)
        return cls(kind=kind, mu=mu, sigma_matrix=Sigma, gamma1=gamma1, gamma2=gamma2, rho_km=rho_km)
