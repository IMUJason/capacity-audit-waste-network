"""Analysis: holding regret, bootstrap intervals, paired differences,
and the expansion-cost trade-off for the capacity ladder.

Reads results/ladder.json (from run_ladder.py), evaluates the fixed
reference layout on identical scenario draws, and writes
results/ladder_analysis.json.

Usage:  python experiments/analyze_ladder.py
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from common import (CVAR_ALPHA, CRF, N_SCENARIOS, RISK_WEIGHT, SEEDS,
                    build_covariance_models, build_instance, fit_forecast)
from capacity_audit.ambiguity.spatiotemporal import generate_lognormal_mixture_scenarios
from capacity_audit.models.dro_facility_location import FacilityLocationProblem
from capacity_audit.models.recourse import solve_nominal_recourse

CERTIFIED = [0, 1, 2, 3, 4, 9, 10, 11, 12, 16, 21, 22, 23, 24, 30, 31]
BETAS = (1.0, 10.0, 100.0)  # multiples of the implied capacity rate, yuan/ton-yr
RESULTS = Path(__file__).resolve().parents[1] / "results"


def risk_objective(problem, y, draws):
    costs = np.array(
        [solve_nominal_recourse(problem, y, d).total_cost for d in draws])
    thr = np.quantile(costs, CVAR_ALPHA, method="higher")
    return (CRF * float(problem.fixed_costs @ y)
            + (1 - RISK_WEIGHT) * costs.mean()
            + RISK_WEIGHT * costs[costs >= thr].mean())


def main() -> None:
    t0 = time.time()
    grid = json.loads((RESULTS / "ladder.json").read_text())
    instance, years, values, coords = build_instance()
    base = FacilityLocationProblem.from_instance(instance)
    trend = fit_forecast(values, years)
    models = build_covariance_models(trend, coords)
    n = instance.n_customers
    y_ref = np.zeros(33)
    y_ref[CERTIFIED] = 1.0

    rows = []
    for r in grid["rows"]:
        problem = replace(base, capacities=base.capacities * r["theta"])
        means = trend.forecast_mean * (1.0 + r["delta"])
        normals = np.random.default_rng(r["seed"]).standard_normal(
            (N_SCENARIOS, n))
        draws, _, _ = generate_lognormal_mixture_scenarios(
            means, models[r["spec"]], N_SCENARIOS, r["seed"], normals)
        hold = risk_objective(problem, y_ref, draws)
        rows.append({
            "theta": r["theta"], "delta": r["delta"], "seed": r["seed"],
            "spec": r["spec"], "optimal_obj": r["objective"],
            "hold_obj": hold,
            "regret_pct": (hold - r["objective"]) / r["objective"] * 100,
            "hamming": len(set(r["open_ids"]) ^ set(CERTIFIED)),
        })
        if len(rows) % 36 == 0:
            print(f"{len(rows)}/{len(grid['rows'])} evaluated ({time.time()-t0:.0f}s)")

    out = {"rows": rows}
    # per-cell bootstrap intervals
    rng = np.random.default_rng(20260908)
    cells = {}
    for theta in sorted({r["theta"] for r in rows}):
        for delta in sorted({r["delta"] for r in rows}):
            regs = np.array([r["regret_pct"] for r in rows
                             if r["theta"] == theta and r["delta"] == delta])
            boots = [rng.choice(regs, len(regs)).mean() for _ in range(5000)]
            cells[f"theta={theta:g}|delta={delta:+.0%}"] = {
                "regret_mean_pct": float(regs.mean()),
                "regret_ci95_pct": [float(np.percentile(boots, 2.5)),
                                    float(np.percentile(boots, 97.5))],
                "regret_max_pct": float(regs.max()),
                "hamming_max": int(max(r["hamming"] for r in rows
                                       if r["theta"] == theta
                                       and r["delta"] == delta)),
            }
    # expansion-cost trade-off (theta > 1)
    cap_total = float(base.capacities.sum())
    tradeoff = {}
    for beta in BETAS:
        table = {}
        for theta in sorted({r["theta"] for r in rows if r["theta"] > 1.0}):
            for delta in sorted({r["delta"] for r in rows}):
                sel = [r for r in rows
                       if r["theta"] == theta and r["delta"] == delta]
                obj = np.mean([r["optimal_obj"] for r in sel])
                cost_pct = beta * (theta - 1.0) * cap_total / obj * 100
                table[f"theta={theta:g}|delta={delta:+.0%}"] = {
                    "regret_mean_pct": float(
                        np.mean([r["regret_pct"] for r in sel])),
                    "expansion_cost_pct_of_obj": cost_pct,
                }
        tradeoff[f"beta={beta:g}"] = table
    out["cell_stats"] = cells
    out["capacity_total_ton_yr"] = cap_total
    out["beta_tradeoff"] = tradeoff
    (RESULTS / "ladder_analysis.json").write_text(json.dumps(out, indent=2))
    print(f"written {RESULTS / 'ladder_analysis.json'}")


if __name__ == "__main__":
    main()
