"""Ride staffing: assign one trained employee per ride, minimizing non-favorite picks."""
import itertools
import random
from heapq import heappush, heappop

import numpy as np
from scipy.optimize import linear_sum_assignment

INF = float("inf")
BIG = 1000  # penalty for an untrained pairing in the Hungarian matrix


class Problem:
    def __init__(self, rides, employees, training, favorite):
        self.rides = rides
        self.employees = employees
        self.training = training
        self.favorite = favorite

    def initial_state(self):
        return (None,) * len(self.rides)  # one slot per ride

    def cost(self, emp, ride):
        return 0 if self.favorite[emp] == ride else 1

    def successors(self, state):
        i = state.index(None)  # always fill the next empty ride
        ride = self.rides[i]
        used = set(state)
        for emp in self.employees:
            if emp not in used and ride in self.training[emp]:
                new = list(state)
                new[i] = emp
                yield tuple(new), self.cost(emp, ride)

    def _remaining(self, state):
        rides = [r for r, e in zip(self.rides, state) if e is None]
        emps = [e for e in self.employees if e not in state]
        return rides, emps

    # ---- heuristics (all admissible) ----
    def h_zero(self, state):
        return 0  # makes A* behave exactly like UCS

    def h_favorite(self, state):
        # a ride costs >= 1 if no free trained employee has it as favorite
        rides, emps = self._remaining(state)
        return sum(
            1 for r in rides
            if not any(self.favorite[e] == r and r in self.training[e] for e in emps)
        )

    def h_hungarian(self, state):
        # exact optimum of the remaining subproblem: strongest possible, costlier per node
        rides, emps = self._remaining(state)
        return hungarian_cost(self, rides, emps)


def hungarian_cost(p, rides, emps):
    """Optimal cost via the Hungarian algorithm. INF if infeasible."""
    if not rides:
        return 0
    if len(emps) < len(rides):
        return INF
    m = np.array([[p.cost(e, r) if r in p.training[e] else BIG for r in rides] for e in emps])
    rows, cols = linear_sum_assignment(m)
    total = m[rows, cols].sum()
    return INF if total >= BIG else int(total)


def search(p, h):
    """A* with heuristic h. Returns dict or None if no valid assignment exists."""
    start = p.initial_state()
    h0 = h(start)
    if h0 == INF:
        return None
    tie = itertools.count()  # tiebreaker so the heap never compares states
    frontier = [(h0, next(tie), 0, start)]
    best = {start: 0}
    expansions = 0
    while frontier:
        _, _, g, s = heappop(frontier)
        if g > best[s]:
            continue  # stale heap entry
        expansions += 1
        if None not in s:
            return {"cost": g, "state": s, "expansions": expansions}
        for succ, step in p.successors(s):
            ng = g + step
            if ng < best.get(succ, INF):
                hv = h(succ)
                if hv == INF:
                    continue  # dead end, prune
                best[succ] = ng
                heappush(frontier, (ng + hv, next(tie), ng, succ))
    return None


def default_instance():
    rides = ["RollerCoaster", "FerrisWheel", "HauntedHouse", "WaterRide", "Carousel", "DropTower", "BumperCars"]
    employees = [f"E{i}" for i in range(1, 11)]
    training = {
        "E1": ["RollerCoaster", "FerrisWheel"],
        "E2": ["FerrisWheel", "HauntedHouse"],
        "E3": ["RollerCoaster", "HauntedHouse", "WaterRide"],
        "E4": ["HauntedHouse", "WaterRide", "Carousel"],
        "E5": ["RollerCoaster", "FerrisWheel", "DropTower"],
        "E6": ["HauntedHouse", "Carousel"],
        "E7": ["WaterRide", "Carousel", "DropTower", "BumperCars"],
        "E8": ["RollerCoaster", "HauntedHouse", "BumperCars"],
        "E9": ["FerrisWheel", "WaterRide"],
        "E10": ["HauntedHouse", "WaterRide", "Carousel", "DropTower", "BumperCars"],
    }
    favorite = {
        "E1": "FerrisWheel", "E2": "HauntedHouse", "E3": "RollerCoaster", "E4": "Carousel",
        "E5": "DropTower", "E6": "Carousel", "E7": "BumperCars", "E8": "RollerCoaster",
        "E9": "WaterRide", "E10": "DropTower",
    }
    return Problem(rides, employees, training, favorite)


def random_instance(n_rides, n_emps, seed=0):
    rng = random.Random(seed)  # seeded so results are reproducible
    rides = [f"Ride{i + 1}" for i in range(n_rides)]
    employees = [f"E{i + 1}" for i in range(n_emps)]
    training = {e: rng.sample(rides, rng.randint(2, min(4, n_rides))) for e in employees}
    favorite = {e: rng.choice(training[e]) for e in employees}  # favorite is always a trained ride
    return Problem(rides, employees, training, favorite)


if __name__ == "__main__":
    import time
    p = default_instance()
    for name, h in [("UCS", p.h_zero), ("A* favorite", p.h_favorite), ("A* hungarian", p.h_hungarian)]:
        t = time.perf_counter()
        r = search(p, h)
        print(f"{name}: cost={r['cost']} expanded={r['expansions']} time={time.perf_counter() - t:.4f}s")
    print("Hungarian baseline cost:", hungarian_cost(p, p.rides, p.employees))