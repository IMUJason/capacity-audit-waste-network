"""Experiment 5: tactical-layer audit via exact subset analysis.

Solves the city-scale dispatch instance exactly (route enumeration +
set partitioning) on the full facility set and on every single- and
pair-removal subset, establishing monotonicity and pricing the tactical
regret of a fixed site set. Also checks the capacity necessary
condition for the larger instance's load-bearing site.

Usage:  python experiments/run_tactical.py
Output: results/tactical.json
"""
from __future__ import annotations

import itertools
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tactical import TacticalInstance, solve_exact

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "tactical.json"


def drop(inst, indices):
    mod = replace(inst, facilities=[f for k, f in enumerate(inst.facilities)
                                    if k not in indices])
    mod.facility_map = {f.id: f for f in mod.facilities}
    mod.customer_map = inst.customer_map
    mod._distance_cache = {}
    return mod


def main() -> None:
    inst = TacticalInstance.load(REPO / "data" / "raw" / "qz_real_1.json")
    m = len(inst.facilities)
    demand = sum(c.demand for c in inst.customers)
    capacity = {f.id: f.capacity for f in inst.facilities}

    rows = []
    configs = [("full", [])]
    configs += [(f"drop{j}", [j]) for j in range(m)]
    configs += [(f"drop{a}{b}", [a, b])
                for a, b in itertools.combinations(range(m), 2)]
    for name, drop_idx in configs:
        mod = inst if not drop_idx else drop(inst, drop_idx)
        res = solve_exact(mod, time_limit=300)
        rows.append({"config": name,
                     "objective": res["metrics"]["objective"],
                     "feasible": res["metrics"]["feasible"],
                     "open": res["metrics"]["open_facilities"]})
        print(f"{name:8s} {res['metrics']['objective']:.4f} "
              f"feasible={res['metrics']['feasible']}")

    full = rows[0]["objective"]
    mono = all(r["objective"] >= full - 1e-6
               for r in rows[1:] if r["feasible"])
    # capacity necessary condition per facility (larger instance)
    cap_check = {}
    big = TacticalInstance.load(REPO / "data" / "raw" / "qz_medium_1.json")
    big_demand = sum(c.demand for c in big.customers)
    for f in big.facilities:
        rest = sum(g.capacity for g in big.facilities if g.id != f.id)
        cap_check[f.id] = {"capacity": f.capacity,
                           "rest_capacity": rest,
                           "removal_infeasible_by_capacity": rest < big_demand}

    OUT.write_text(json.dumps({
        "run_at": datetime.now().isoformat(),
        "instance": inst.name,
        "full_optimum": full,
        "monotonicity_holds": mono,
        "rows": rows,
        "capacity_necessary_condition": {
            "instance": big.name, "demand": big_demand, "per_facility": cap_check},
    }, indent=2))
    print(f"full optimum {full:.4f}; monotonicity: {mono}")
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
