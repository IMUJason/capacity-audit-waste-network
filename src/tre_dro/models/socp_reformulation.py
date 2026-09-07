"""Closed form for a linear loss over the configured moment ambiguity set.

This identity is exact for one fixed linear functional. Applying it to a single
dual vector of a piecewise-linear recourse function gives a valid lower bound,
not an exact reformulation of the full two-stage DRO problem.

Core identity (D-Y Theorem 1, linear functional):
    sup_{P∈F(μ,Σ,γ₁,γ₂)} E_P[πᵀ d] = πᵀ μ + Ω ‖Σ^{1/2} π‖₂
    where kappa = sqrt(min(gamma1, gamma2)).

This is verified by brute-force Monte Carlo sampling over admissible means in
the ellipsoid (test_socp_dual.py).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..ambiguity.ambiguity_set import AmbiguitySet


def worst_case_closed_form(
    pi: np.ndarray, amb: AmbiguitySet
) -> tuple[float, dict[str, Any]]:
    """Compute πᵀ μ + Ω ‖Σ^{1/2} π‖₂ and return a diagnostic dict.

    Parameters
    ----------
    pi : (n,) shape — linear functional coefficients (transport dual variables).
    amb : AmbiguitySet

    Returns
    -------
    value : float — supremum over P∈F.
    diag : dict — components: nominal, robust_term, omega, robust_norm.
    """
    pi_arr = np.asarray(pi, float)
    nominal = float(pi_arr @ amb.mu)
    sigma_half = amb.sigma_half
    from ..ambiguity.covariance import robust_norm

    rn = robust_norm(sigma_half, pi_arr)
    omega = amb.omega
    robust_term = omega * rn
    return nominal + robust_term, {
        "nominal": nominal,
        "omega": omega,
        "robust_norm": rn,
        "robust_term": robust_term,
        "kind": amb.kind,
        "gamma1": amb.gamma1,
        "gamma2": amb.gamma2,
        "rho_km": amb.rho_km,
    }
