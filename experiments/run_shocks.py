"""Experiment 4: physical shocks to the network.

Seven tested shocks: closing each marginal facility, both together,
capacity tightened by 25%, a marginal closure under 20% demand growth,
and two load-bearing closures (largest-demand city site and a
dense-cluster site). For each shock the script reports the pre-shock
objective, the post-shock hold objective (fixed degraded network), the
post-shock re-optimized objective, the adaptive regret (hold vs
re-opt), the shock cost, and service statistics.

Usage:  python experiments/run_shocks.py [--smoke]
Output: results/shocks.json
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

from common import (CERTIFIED, CVAR_ALPHA, CRF, RISK_WEIGHT,
                    build_covariance_models, build_instance, fit_forecast)
from capacity_audit.ambiguity.spatiotemporal import (
    generate_lognormal_mixture_scenarios,
)
from capacity_audit.models.dro_facility_location import FacilityLocationProblem
from capacity_audit.models.recourse import solve_nominal_recourse
from capacity_audit.solvers.stochastic_cvar import solve_mean_cvar_saa

SEED = 20260714
OUT = Path(__file__).resolve().parents[1] / "results" / "shocks.json"


def evaluate(problem, y, draws):
    costs, unmet, served = [], [], []
    for d in draws:
        sol = solve_nominal_recourse(problem, y, d)
        costs.append(sol.total_cost)
        unmet.append(float(sol.unmet.sum()))
        served.append(float(d.sum() - sol.unmet.sum()))
    costs = np.array(costs)
    thr = np.quantile(costs, CVAR_ALPHA, method="higher")
    return {
        "objective": (CRF * float(problem.fixed_costs @ y)
                      + (1 - RISK_WEIGHT) * costs.mean()
                      + RISK_WEIGHT * costs[costs >= thr].mean()),
        "service": float(np.mean(np.array(served)
                        / (np.array(served) + np.array(unmet)))),
    }


def main() -> None:
    smoke = "--smoke" in sys.argv
    n_scen = 60 if smoke else 600
    t0 = time.time()
    OUT.parent.mkdir(exist_ok=True)
    instance, years, values, coords = build_instance()
    base = FacilityLocationProblem.from_instance(instance)
    trend = fit_forecast(values, years)
    cov = build_covariance_models(trend, coords)["Independent"]
    draws0, _, _ = generate_lognormal_mixture_scenarios(
        trend.forecast_mean, cov, n_scen, SEED,
        np.random.default_rng(SEED).standard_normal((n_scen, 40)))
    draws_g, _, _ = generate_lognormal_mixture_scenarios(
        trend.forecast_mean * 1.2, cov, n_scen, SEED,
        np.random.default_rng(SEED).standard_normal((n_scen, 40)))
    y_cert = np.zeros(33)
    y_cert[CERTIFIED] = 1.0

    def closed(sites):
        caps = base.capacities.copy()
        for j in sites:
            caps[j] = 0.0
        return replace(base, capacities=caps)

    shocks = {
        "close_marginal_a": (closed([30]), [j for j in CERTIFIED if j != 30], draws0),
        "close_marginal_b": (closed([2]), [j for j in CERTIFIED if j != 2], draws0),
        "close_both_marginals": (closed([30, 2]), [j for j in CERTIFIED if j not in (30, 2)], draws0),
        "capacity_x075": (replace(base, capacities=base.capacities * 0.75), CERTIFIED, draws0),
        "marginal_closure_growth": (closed([30]), [j for j in CERTIFIED if j != 30], draws_g),
        "close_loadbearing_a": (closed([0]), [j for j in CERTIFIED if j != 0], draws0),
        "close_loadbearing_b": (closed([9]), [j for j in CERTIFIED if j != 9], draws0),
    }

    pre = evaluate(base, y_cert, draws0)
    rows = {"pre_shock": pre}
    for name, (prob, hold_ids, draws) in shocks.items():
        y_hold = np.zeros(33)
        y_hold[hold_ids] = 1.0
        hold = evaluate(prob, y_hold, draws)
        res = solve_mean_cvar_saa(
            prob, draws, risk_weight=RISK_WEIGHT, cvar_alpha=CVAR_ALPHA,
            fixed_cost_multiplier=CRF, time_limit=1800)
        rows[name] = {
            "post_hold": hold, "post_reopt_obj": float(res.objective),
            "adaptive_regret_pct": (hold["objective"] - res.objective)
                                   / res.objective * 100,
            "shock_cost_pct": (hold["objective"] - pre["objective"])
                              / pre["objective"] * 100,
            "reopt_open": int(res.y_open.sum()),
        }
        print(f"{name}: adaptive {rows[name]['adaptive_regret_pct']:+.5f}% "
              f"shock {rows[name]['shock_cost_pct']:+.3f}% "
              f"({time.time()-t0:.0f}s)")

    rows["design"] = {"n_scenarios": n_scen, "seed": SEED, "smoke": smoke}
    OUT.write_text(json.dumps(rows, indent=2))
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
