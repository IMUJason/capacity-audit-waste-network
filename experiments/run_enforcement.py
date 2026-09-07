"""Experiment 2: enforcement x capacity grid with decision metrics.

Crosses unmet-penalty levels {300, 500, 750, 1000} yuan/ton with
capacity rungs {0.75, 1.0, 1.25}, four covariance specifications,
three seeds (144 solves), and reports for every cell the re-optimized
network plus the fixed-layout holding regret, service rate, and mean
unmet tonnage.

Usage:  python experiments/run_enforcement.py
Output: results/enforcement.json
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
from tre_dro.ambiguity.spatiotemporal import generate_lognormal_mixture_scenarios
from tre_dro.models.dro_facility_location import FacilityLocationProblem
from tre_dro.models.recourse import solve_nominal_recourse
from tre_dro.solvers.stochastic_cvar import solve_mean_cvar_saa

PENALTIES = (300, 500, 750, 1000)
THETAS = (0.75, 1.0, 1.25)
CERTIFIED = [0, 1, 2, 3, 4, 9, 10, 11, 12, 16, 21, 22, 23, 24, 30, 31]
OUT = Path(__file__).resolve().parents[1] / "results" / "enforcement.json"


def main() -> None:
    t0 = time.time()
    OUT.parent.mkdir(exist_ok=True)
    instance, years, values, coords = build_instance()
    base = FacilityLocationProblem.from_instance(instance)
    trend = fit_forecast(values, years)
    models = build_covariance_models(trend, coords)
    n = instance.n_customers
    y_ref = np.zeros(33)
    y_ref[CERTIFIED] = 1.0

    rows = []
    for penalty in PENALTIES:
        for theta in THETAS:
            problem = replace(base, capacities=base.capacities * theta,
                              unmet_penalty=float(penalty))
            for seed in SEEDS:
                normals = np.random.default_rng(seed).standard_normal(
                    (N_SCENARIOS, n))
                for spec, cov in models.items():
                    draws, _, _ = generate_lognormal_mixture_scenarios(
                        trend.forecast_mean, cov, N_SCENARIOS, seed, normals)
                    res = solve_mean_cvar_saa(
                        problem, draws, risk_weight=RISK_WEIGHT,
                        cvar_alpha=CVAR_ALPHA, fixed_cost_multiplier=CRF,
                        time_limit=1800)
                    costs, unmet, served = [], [], []
                    for d in draws:
                        sol = solve_nominal_recourse(problem, y_ref, d)
                        costs.append(sol.total_cost)
                        unmet.append(float(sol.unmet.sum()))
                        served.append(float(d.sum() - sol.unmet.sum()))
                    costs = np.array(costs)
                    thr = np.quantile(costs, CVAR_ALPHA, method="higher")
                    hold = (CRF * float(problem.fixed_costs @ y_ref)
                            + (1 - RISK_WEIGHT) * costs.mean()
                            + RISK_WEIGHT * costs[costs >= thr].mean())
                    rows.append({
                        "penalty": penalty, "theta": theta, "seed": seed,
                        "spec": spec,
                        "reopt_obj": float(res.objective),
                        "reopt_open": int(res.y_open.sum()),
                        "hamming": len(set(
                            np.flatnonzero(res.y_open > 0.5).tolist())
                            ^ set(CERTIFIED)),
                        "hold_obj": hold,
                        "hold_regret_pct": (hold - res.objective)
                                           / res.objective * 100,
                        "service_rate": float(np.mean(
                            np.array(served)
                            / (np.array(served) + np.array(unmet)))),
                        "unmet_ton_mean": float(np.mean(unmet)),
                    })
            print(f"penalty={penalty} theta={theta}: {len(rows)} rows "
                  f"({time.time()-t0:.0f}s)")

    OUT.write_text(json.dumps({
        "run_at": datetime.now().isoformat(),
        "design": {"penalties": list(PENALTIES), "thetas": list(THETAS),
                   "specs": list(models), "seeds": list(SEEDS),
                   "n_scenarios": N_SCENARIOS},
        "rows": rows,
    }, indent=2))
    print(f"written {OUT} ({len(rows)} solves)")


if __name__ == "__main__":
    main()
