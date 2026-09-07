"""Linear moment-bound coefficient kappa = sqrt(min(gamma1, gamma2)).

This module replaces the old misuse at ``spatial_ambiguity_set.py:238`` where
``omega = scipy.stats.norm.ppf(confidence)`` treated a *normal z-quantile* as a
*moment-confidence-ellipsoid radius*. Per Delage & Ye (2010, Thm. 1), the
worst-case expected linear loss over the moment ambiguity set is

    sup_{P∈F} E_P[πᵀ d] = πᵀ μ + sqrt(min(γ₁,γ₂)) · ‖Σ^{1/2} π‖

for the centered second-moment set used here. It is not a Gaussian quantile.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def omega_from_gamma(gamma1: float, gamma2: float) -> float:
    """Worst-case linear-loss coefficient  κ = √(min(γ₁, γ₂)).

    Under the Delage-Ye (2010) joint moment ambiguity set
        F = {P: (E[d]-μ)ᵀΣ⁻¹(E[d]-μ) ≤ γ₁,
                 E[(d-μ)(d-μ)ᵀ] ⪯ γ₂ Σ}
    both constraints reduce to ellipsoidal bounds on the mean-deviation
    m = E[d] − μ  via  mᵀ Σ⁻¹ m ≤ min(γ₁, γ₂).  The worst-case linear loss
    is therefore  πᵀμ + √(min(γ₁,γ₂)) · ‖Σ^{1/2} π‖.

    The coupling of γ₁ and γ₂ is a MIN (whichever binds first), not a product.
    """
    return float(np.sqrt(min(gamma1, gamma2)))


def recommend_gamma(confidence: float, n_samples: int, dim: int) -> tuple[float, float]:
    """Data-driven (γ₁, γ₂) for a desired confidence level (Delage & Ye 2010).

    γ₁ sizes the mean-deviation ellipsoid via a χ² bound on ‖μ̂−μ‖_{Σ⁻¹}²;
    γ₂ inflates the covariance against second-moment estimation error.

    Returned values are the *standard calibrated choices* used in the paper's
    appendix; users may also set (γ₁, γ₂) directly for controlled experiments.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0,1)")
    n = max(n_samples, 1)
    # Mean ellipsoid: P(‖μ̂-μ‖² ≤ γ₁) ≥ confidence  →  γ₁ ≈ χ²_{dim,conf} / n
    gamma1 = float(stats.chi2.ppf(confidence, df=dim) / n)
    # Covariance inflation (conservative Delage-Ye form, §3)
    delta = 1.0 - confidence
    gamma2 = float(1.0 / max(1.0 - np.sqrt(2.0 * np.log(2.0 / delta) / n), 1e-3))
    return gamma1, gamma2
