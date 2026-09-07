"""Experiment 1: capacity-availability ladder.

Solves the strategic model across capacity rungs theta in
{0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.4, 1.5}, demand shifts
{-20%, 0, +20}%, four covariance specifications, three seeds
(288 solves) and records the re-optimized layout at every cell.

Usage:  python experiments/run_ladder.py
Output: results/ladder.json
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from common import (CVAR_ALPHA, CRF, N_SCENARIOS, RISK_WEIGHT, SEEDS,
                    build_covariance_models, build_instance, fit_forecast)
from capacity_audit.ambiguity.spatiotemporal import generate_lognormal_mixture_scenarios
from capacity_audit.models.dro_facility_location import FacilityLocationProblem
from capacity_audit.solvers.stochastic_cvar import solve_mean_cvar_saa

THETAS = (0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.4, 1.5)
DELTAS = (-0.20, 0.0, 0.20)
OUT = Path(__file__).resolve().parents[1] / "results" / "ladder.json"


def main() -> None:
    t0 = time.time()
    OUT.parent.mkdir(exist_ok=True)
    instance, years, values, coords = build_instance()
    base = FacilityLocationProblem.from_instance(instance)
    trend = fit_forecast(values, years)
    models = build_covariance_models(trend, coords)
    n = instance.n_customers

    rows = []
    for theta in THETAS:
        problem = replace(base, capacities=base.capacities * theta)
        for delta in DELTAS:
            means = trend.forecast_mean * (1.0 + delta)
            for seed in SEEDS:
                normals = np.random.default_rng(seed).standard_normal(
                    (N_SCENARIOS, n))
                for spec, cov in models.items():
                    draws, _, _ = generate_lognormal_mixture_scenarios(
                        means, cov, N_SCENARIOS, seed, normals)
                    res = solve_mean_cvar_saa(
                        problem, draws, risk_weight=RISK_WEIGHT,
                        cvar_alpha=CVAR_ALPHA, fixed_cost_multiplier=CRF,
                        time_limit=1800)
                    rows.append({
                        "theta": theta, "delta": delta, "seed": seed,
                        "spec": spec, "status": res.status,
                        "objective": float(res.objective),
                        "open_ids": np.flatnonzero(res.y_open > 0.5).tolist(),
                    })
        print(f"theta={theta} done ({time.time()-t0:.0f}s)")

    OUT.write_text(json.dumps({
        "run_at": datetime.now().isoformat(),
        "design": {"thetas": list(THETAS), "deltas": list(DELTAS),
                   "specs": list(models), "seeds": list(SEEDS),
                   "n_scenarios": N_SCENARIOS},
        "rows": rows,
    }, indent=2))
    print(f"written {OUT} ({len(rows)} solves)")


if __name__ == "__main__":
    main()
