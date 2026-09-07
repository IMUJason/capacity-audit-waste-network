"""Experiment 3: fine-grained demand sweep with conservative spatial
redistributions.

Eleven demand levels (+/-2 to +/-20%) and three spatial-redistribution
strengths in two independent directions each (demand-weighted
conservation asserted per forecast year), four covariance
specifications, three seeds.

Usage:  python experiments/run_demand_boundary.py [--smoke]
Output: results/demand_boundary.json
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

SEEDS = (20260714, 20260731, 20260801)
LEVELS = (-0.20, -0.15, -0.10, -0.05, -0.02, 0.0, 0.02, 0.05,
          0.10, 0.15, 0.20)
SPATIAL = (0.10, 0.20, 0.30)
DIR_BASES = (900, 910)
OUT = Path(__file__).resolve().parents[1] / "results" / "demand_boundary.json"


def conservative_multiplier(mu_row, strength, seed):
    z = np.random.default_rng(seed).standard_normal(len(mu_row))
    w = 1.0 + strength * z
    w = w * (mu_row.sum() / (mu_row * w).sum())
    assert abs((mu_row * w).sum() - mu_row.sum()) / mu_row.sum() < 1e-12
    return w


def risk_objective(problem, y, draws):
    costs = np.array(
        [solve_nominal_recourse(problem, y, d).total_cost for d in draws])
    thr = np.quantile(costs, CVAR_ALPHA, method="higher")
    return (CRF * float(problem.fixed_costs @ y)
            + (1 - RISK_WEIGHT) * costs.mean()
            + RISK_WEIGHT * costs[costs >= thr].mean())


def main() -> None:
    smoke = "--smoke" in sys.argv
    n_scen, seeds = (60, SEEDS[:1]) if smoke else (600, SEEDS)
    t0 = time.time()
    OUT.parent.mkdir(exist_ok=True)
    instance, years, values, coords = build_instance()
    problem = FacilityLocationProblem.from_instance(instance)
    trend = fit_forecast(values, years)
    models = build_covariance_models(trend, coords)
    base = trend.forecast_mean
    n = instance.n_customers
    y_ref = np.zeros(33)
    y_ref[CERTIFIED] = 1.0

    rows = []
    conditions = [("level", d, None) for d in LEVELS]
    conditions += [("spatial", None, s) for s in SPATIAL for _ in (0,)]
    for seed in seeds:
        normals = np.random.default_rng(seed).standard_normal((n_scen, n))
        for kind, delta, s in conditions:
            for direction in ((DIR_BASES if kind == "spatial" else (0,))):
                if kind == "level":
                    means = base * (1.0 + delta)
                else:
                    mult = np.vstack([
                        conservative_multiplier(base[yy], s, direction)
                        for yy in range(base.shape[0])])
                    means = base * mult
                for spec, cov in models.items():
                    if smoke and spec != "Independent":
                        continue
                    draws, _, _ = generate_lognormal_mixture_scenarios(
                        means, cov, n_scen, seed, normals)
                    res = solve_mean_cvar_saa(
                        problem, draws, risk_weight=RISK_WEIGHT,
                        cvar_alpha=CVAR_ALPHA, fixed_cost_multiplier=CRF,
                        time_limit=1800)
                    ids = set(np.flatnonzero(res.y_open > 0.5).tolist())
                    hold = risk_objective(problem, y_ref, draws)
                    rows.append({
                        "kind": kind, "delta": delta,
                        "spatial_strength": s if kind == "spatial" else None,
                        "direction": direction if kind == "spatial" else None,
                        "seed": seed, "spec": spec,
                        "open_count": len(ids),
                        "hamming": len(ids ^ set(CERTIFIED)),
                        "regret_pct": (hold - res.objective)
                                      / res.objective * 100,
                    })
        print(f"seed {seed} done ({time.time()-t0:.0f}s)")

    OUT.write_text(json.dumps({
        "run_at": datetime.now().isoformat(),
        "design": {"levels": list(LEVELS), "spatial": list(SPATIAL),
                   "directions": list(DIR_BASES), "seeds": list(seeds),
                   "n_scenarios": n_scen, "smoke": smoke},
        "rows": rows,
    }, indent=2))
    print(f"written {OUT} ({len(rows)} solves)")


if __name__ == "__main__":
    main()
