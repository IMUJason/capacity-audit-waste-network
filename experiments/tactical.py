"""City-scale tactical module: instance IO, evaluation, and the exact
route-enumeration/set-partitioning solver used for the tactical audit.

The instance family is a daily collection-and-dispatch problem built
from regulatory e-manifest records (hotspot weights and site list);
vehicle routes serve time windows, a central restricted zone slows
travel inside a time window, disposal sites have capacities and fixed
costs. The exact solver enumerates, per facility, the cheapest
feasible ordering of every customer subset up to a size cap and solves
the resulting set-partitioning problem to optimality.
"""
from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from docplex.mp.model import Model


def haversine_km(lon1, lat1, lon2, lat2) -> float:
    radius = 6371.0
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    a = (math.sin((lat2 - lat1) / 2.0) ** 2
         + math.cos(lat1) * math.cos(lat2)
         * math.sin((lon2 - lon1) / 2.0) ** 2)
    return 2.0 * radius * math.asin(math.sqrt(a))


@dataclass
class Customer:
    id: int
    name: str
    x: float
    y: float
    demand: int
    earliest: int
    latest: int
    service_time: int
    hotspot: str
    hotspot_weight: float
    source_project: str


@dataclass
class Facility:
    id: int
    name: str
    x: float
    y: float
    capacity: int
    fixed_cost: float
    source_site: str
    site_weight: float


@dataclass
class TacticalInstance:
    name: str
    vehicle_capacity: int
    vehicle_fixed_cost: float
    transport_cost_per_km: float
    max_route_minutes: int
    day_start: int
    day_end: int
    average_speed_kmph: float
    restricted_center: Dict[str, float]
    restricted_radius_km: float
    restricted_window: Tuple[int, int]
    restricted_penalty_factor: float
    customers: List[Customer] = field(default_factory=list)
    facilities: List[Facility] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "TacticalInstance":
        d = json.loads(Path(path).read_text())
        inst = cls(
            name=d["name"],
            vehicle_capacity=int(d["vehicle_capacity"]),
            vehicle_fixed_cost=float(d["vehicle_fixed_cost"]),
            transport_cost_per_km=float(d["transport_cost_per_km"]),
            max_route_minutes=int(d["max_route_minutes"]),
            day_start=int(d["day_start"]),
            day_end=int(d["day_end"]),
            average_speed_kmph=float(d["average_speed_kmph"]),
            restricted_center=d["restricted_center"],
            restricted_radius_km=float(d["restricted_radius_km"]),
            restricted_window=tuple(d["restricted_window"]),
            restricted_penalty_factor=float(d["restricted_penalty_factor"]),
            customers=[Customer(**c) for c in d["customers"]],
            facilities=[Facility(**f) for f in d["facilities"]],
        )
        inst.customer_map = {c.id: c for c in inst.customers}
        inst.facility_map = {f.id: f for f in inst.facilities}
        inst._distance_cache = {}
        return inst

    def distance(self, kind_a, id_a, kind_b, id_b) -> float:
        key = (kind_a, id_a, kind_b, id_b)
        if key in self._distance_cache:
            return self._distance_cache[key]
        a = (self.customer_map if kind_a == "c" else self.facility_map)[id_a]
        b = (self.customer_map if kind_b == "c" else self.facility_map)[id_b]
        value = haversine_km(a.x, a.y, b.x, b.y)
        self._distance_cache[key] = value
        self._distance_cache[(kind_b, id_b, kind_a, id_a)] = value
        return value

    def travel_time(self, kind_a, id_a, kind_b, id_b, departure) -> float:
        dist = self.distance(kind_a, id_a, kind_b, id_b)
        minutes = dist / max(self.average_speed_kmph, 1e-6) * 60.0
        a = (self.customer_map if kind_a == "c" else self.facility_map)[id_a]
        b = (self.customer_map if kind_b == "c" else self.facility_map)[id_b]
        mid_lon = (a.x + b.x) / 2.0
        mid_lat = (a.y + b.y) / 2.0
        inside = haversine_km(mid_lon, mid_lat,
                              self.restricted_center["x"],
                              self.restricted_center["y"]) \
            <= self.restricted_radius_km
        in_window = (self.restricted_window[0] <= departure
                     <= self.restricted_window[1])
        if inside and in_window:
            minutes *= self.restricted_penalty_factor
        return minutes

    def route_metrics(self, facility_id, customers: Sequence[int]) -> Dict:
        t = float(self.day_start)
        distance = wait = lateness = load = 0.0
        prev_kind, prev_id = "f", facility_id
        for cid in customers:
            c = self.customer_map[cid]
            travel = self.travel_time(prev_kind, prev_id, "c", cid, t)
            distance += self.distance(prev_kind, prev_id, "c", cid)
            t += travel
            if t < c.earliest:
                wait += c.earliest - t
                t = float(c.earliest)
            if t > c.latest:
                lateness += t - c.latest
            t += c.service_time
            load += c.demand
            prev_kind, prev_id = "c", cid
        if customers:
            travel = self.travel_time(prev_kind, prev_id, "f", facility_id, t)
            distance += self.distance(prev_kind, prev_id, "f", facility_id)
            t += travel
        return {"distance": distance, "duration": t - self.day_start,
                "lateness": lateness, "wait": wait, "load": load}

    def evaluate(self, solution: Dict) -> Dict:
        fixed = vehicle = transport = 0.0
        travel = lateness = overload = overtime = fac_over = 0.0
        usage = {f.id: 0 for f in self.facilities}
        n_routes = 0
        for route in solution["routes"]:
            if not route["customers"]:
                continue
            m = self.route_metrics(route["facility_id"], route["customers"])
            usage[route["facility_id"]] += m["load"]
            vehicle += self.vehicle_fixed_cost
            transport += m["distance"] * self.transport_cost_per_km
            travel += m["distance"]
            lateness += m["lateness"]
            overload += max(m["load"] - self.vehicle_capacity, 0)
            overtime += max(m["duration"] - self.max_route_minutes, 0)
            n_routes += 1
        for fid, load in usage.items():
            if load > 0:
                fixed += self.facility_map[fid].fixed_cost
            fac_over += max(load - self.facility_map[fid].capacity, 0)
        penalty = (10000.0 * _missing(self, solution)
                   + 8000.0 * _duplicated(solution)
                   + 1800.0 * overload + 1200.0 * fac_over
                   + 250.0 * overtime + 180.0 * lateness)
        return {"objective": fixed + vehicle + transport + penalty,
                "fixed_cost": fixed, "vehicle_cost": vehicle,
                "transport_cost": transport, "travel_distance": travel,
                "penalty": penalty, "routes": n_routes,
                "feasible": penalty < 1e-6,
                "open_facilities": sum(1 for v in usage.values() if v > 0)}


def _missing(inst, solution):
    assigned = {c for r in solution["routes"] for c in r["customers"]}
    return sum(1 for c in inst.customers if c.id not in assigned)


def _duplicated(solution):
    seen, dup = set(), set()
    for r in solution["routes"]:
        for c in r["customers"]:
            if c in seen:
                dup.add(c)
            seen.add(c)
    return len(dup)


def enumerate_feasible_routes(inst: TacticalInstance, facility_id: int,
                              max_subset_size: int = 5) -> List[Dict]:
    ids = [c.id for c in inst.customers]
    routes, rid = [], 0
    for size in range(1, min(max_subset_size, len(ids)) + 1):
        for subset in itertools.combinations(ids, size):
            load = sum(inst.customer_map[c].demand for c in subset)
            if load > inst.vehicle_capacity:
                continue
            best_seq, best_dist = None, None
            for order in itertools.permutations(subset):
                m = inst.route_metrics(facility_id, order)
                if (m["load"] <= inst.vehicle_capacity
                        and m["duration"] <= inst.max_route_minutes
                        and m["lateness"] <= 1e-9
                        and (best_dist is None or m["distance"] < best_dist)):
                    best_dist, best_seq = m["distance"], list(order)
            if best_seq is None:
                continue
            routes.append({
                "route_id": f"r_{facility_id}_{rid}",
                "facility_id": facility_id, "customers": best_seq,
                "load": load, "distance": best_dist,
                "route_cost": inst.vehicle_fixed_cost
                + best_dist * inst.transport_cost_per_km,
            })
            rid += 1
    return routes


def solve_exact(inst: TacticalInstance, time_limit: int = 300) -> Dict:
    pool = []
    for f in inst.facilities:
        pool.extend(enumerate_feasible_routes(inst, f.id))
    cover = {c.id: [] for c in inst.customers}
    by_fac = {f.id: [] for f in inst.facilities}
    for r in pool:
        by_fac[r["facility_id"]].append(r)
        for cid in r["customers"]:
            cover[cid].append(r)
    model = Model(name=f"tactical_exact_{inst.name}")
    model.parameters.timelimit = time_limit
    model.parameters.mip.tolerances.mipgap = 0.0
    y = {f.id: model.binary_var(name=f"y_{f.id}") for f in inst.facilities}
    x = {r["route_id"]: model.binary_var(name=r["route_id"]) for r in pool}
    for cid, rs in cover.items():
        if not rs:
            raise RuntimeError(f"no feasible route covers customer {cid}")
        model.add_constraint(model.sum(x[r["route_id"]] for r in rs) == 1)
    for f in inst.facilities:
        for r in by_fac[f.id]:
            model.add_constraint(x[r["route_id"]] <= y[f.id])
        if by_fac[f.id]:
            model.add_constraint(
                model.sum(r["load"] * x[r["route_id"]]
                          for r in by_fac[f.id]) <= f.capacity * y[f.id])
    model.minimize(
        model.sum(f.fixed_cost * y[f.id] for f in inst.facilities)
        + model.sum(r["route_cost"] * x[r["route_id"]] for r in pool))
    sol = model.solve(log_output=False)
    chosen = [r for r in pool if x[r["route_id"]].solution_value > 0.5]
    solution = {"routes": [{"facility_id": r["facility_id"],
                            "customers": list(r["customers"])}
                           for r in chosen]}
    return {"status": sol.solve_status.name,
            "metrics": inst.evaluate(solution),
            "solution": solution, "route_pool_size": len(pool)}
