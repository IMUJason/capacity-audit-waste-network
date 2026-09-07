"""Dual-linearized moment bounds for capacitated recourse.

For a fixed facility decision y and its nominal recourse dual π = (α, β), the
    expression lower-bounds the worst-case expected recourse:

    theta >= alpha' mu + kappa ||Sigma^(1/2) alpha|| + lambda' M y

Re-activates and generalises ``robust_cuts.compute_robust_norm`` (which was
written correctly but disabled at benders_dro.py:548) from diagonal Σ to full
spatial covariance.
"""
from __future__ import annotations

import numpy as np

from ..ambiguity.ambiguity_set import AmbiguitySet


def robust_optimality_cut(
    pi_demand: np.ndarray,       # (n,) demand duals α_i
    pi_capacity: np.ndarray,     # (m,) signed capacity duals lambda_j <= 0
    capacities: np.ndarray,      # (m,) facility capacities
    y_ref: np.ndarray,           # (m,) current open decisions
    amb: AmbiguitySet,
) -> tuple[float, dict]:
    """Return constant and coefficients of a valid dual lower bound.

    The cut is of the form

        theta >= alpha' mu + kappa ||Sigma^(1/2) alpha||
                 + sum_j lambda_j M_j y_j,

    where lambda_j uses the minimization-LP sign convention and is therefore
    non-positive. For piecewise-linear recourse this is a lower bound obtained
    from one dual affine piece; it is not an exact worst-case oracle.
    """
    # Simplified per-customer cut: θ ≥ ∑_i α_i μ_i + Ω‖Σ^{1/2}α‖ − Σ_j β_j M_j y_j_ref
    # where y_j_ref is the current first-stage variable.
    # Actually, Benders cut: θ ≥ Q(y_ref) + ∇Q(y_ref)·(y − y_ref).
    # For linear ∇Q(y_ref) = -β_j M_j, cut is θ ≥ Q(y_ref) − Σ_j β_j M_j (y_j − y_ref_j)
    # = (Q(y_ref) + Σ_j β_j M_j y_ref_j) − Σ_j β_j M_j y_j.
    # The constant part is πᵀ μ + Ω‖Σ^{1/2}α‖ + Σ_j β_j M_j y_ref_j.
    pi_demand_arr = np.asarray(pi_demand, float)
    pi_cap_arr = np.asarray(pi_capacity, float)

    from ..ambiguity.covariance import robust_norm
    from .socp_reformulation import worst_case_closed_form

    wc_val, wc_diag = worst_case_closed_form(pi_demand_arr, amb)

    if np.any(pi_cap_arr > 1e-7):
        raise ValueError("capacity duals must use the signed <= 0 convention")
    coeffs = pi_cap_arr * np.asarray(capacities, float)
    cap_contrib = float(np.sum(coeffs * np.asarray(y_ref, float)))

    components = {
        "worst_case_cost": wc_val,
        "nominal": wc_diag["nominal"],
        "robust_term": wc_diag["robust_term"],
        "capacity_contrib": cap_contrib,
        "cut_constant": wc_val,
        "cut_coefficients": coeffs.tolist(),
        "omega": wc_diag["omega"],
        "kind": wc_diag["kind"],
    }
    return components["cut_constant"], components
