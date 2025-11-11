from heapq import heappush, heappop
import time

RIDES = ["RollerCoaster", "FerrisWheel", "HauntedHouse", "WaterRide", "Carousel", "DropTower", "BumperCars"]
EMPLOYEES = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10"]

TRAINING = {
    "E1": ["RollerCoaster", "FerrisWheel"],
    "E2": ["FerrisWheel", "HauntedHouse"],
    "E3": ["RollerCoaster", "HauntedHouse", "WaterRide"],
    "E4": ["HauntedHouse", "WaterRide", "Carousel"],
    "E5": ["RollerCoaster", "FerrisWheel", "DropTower"],
    "E6": ["HauntedHouse", "Carousel"],
    "E7": ["WaterRide", "Carousel", "DropTower", "BumperCars"],
    "E8": ["RollerCoaster", "HauntedHouse", "BumperCars"],
    "E9": ["FerrisWheel", "WaterRide"],
    "E10": ["HauntedHouse", "WaterRide", "Carousel", "DropTower", "BumperCars"]
}

FAVORITE = {
    "E1": "FerrisWheel",
    "E2": "HauntedHouse",
    "E3": "RollerCoaster",
    "E4": "Carousel",
    "E5": "DropTower",
    "E6": "Carousel",
    "E7": "BumperCars",
    "E8": "RollerCoaster",
    "E9": "WaterRide",
    "E10": "DropTower"
}

def initial_state():
    return tuple([None for _ in RIDES])

def is_goal(state):
    return None not in state

def assigned_employees(state):
    return {e for e in state if e is not None}

def assignment_cost(emp, ride):
    return 0 if FAVORITE[emp] == ride else 1

def successors(state):
    next_index = state.index(None)
    ride = RIDES[next_index]
    used = assigned_employees(state)
    for emp in EMPLOYEES:
        if emp not in used and ride in TRAINING[emp]:
            new_state = list(state)
            new_state[next_index] = emp
            yield tuple(new_state), assignment_cost(emp, ride), (emp, ride)

def heuristic(state):
    remaining_rides = [RIDES[i] for i, emp in enumerate(state) if emp is None]
    remaining_emps = [e for e in EMPLOYEES if e not in assigned_employees(state)]
    cost_estimate = 0
    for ride in remaining_rides:
        if not any(FAVORITE[e] == ride for e in remaining_emps if ride in TRAINING[e]):
            cost_estimate += 1
    return cost_estimate

def uniform_cost_search(start):
    frontier = []
    heappush(frontier, (0, start))
    cost_so_far = {start: 0}
    expansions = 0

    while frontier:
        g, state = heappop(frontier)
        expansions += 1
        if is_goal(state):
            return {
                'algorithm': 'UCS',
                'cost': g,
                'state': state,
                'expansions': expansions
            }

        for succ, step_cost, action in successors(state):
            new_g = g + step_cost
            if succ not in cost_so_far or new_g < cost_so_far[succ]:
                cost_so_far[succ] = new_g
                heappush(frontier, (new_g, succ))


def a_star_search(start):
    frontier = []
    g_start = 0
    f_start = g_start + heuristic(start)
    heappush(frontier, (f_start, g_start, start))
    cost_so_far = {start: 0}
    expansions = 0

    while frontier:
        f, g, state = heappop(frontier)
        expansions += 1
        if is_goal(state):
            return {
                'algorithm': 'A*',
                'cost': g,
                'state': state,
                'expansions': expansions
            }

        for succ, step_cost, action in successors(state):
            new_g = g + step_cost
            if succ not in cost_so_far or new_g < cost_so_far[succ]:
                cost_so_far[succ] = new_g
                f_new = new_g + heuristic(succ)
                heappush(frontier, (f_new, new_g, succ))

if __name__ == "__main__":
    start = initial_state()

    
    start_time = time.time()
    result_ucs = uniform_cost_search(start)
    ucs_time = round(time.time() - start_time, 4)
    print(
        f"UCS finished with total cost {result_ucs['cost']}, "
        f"state: {result_ucs['state']}, "
        f"nodes expanded: {result_ucs['expansions']}, "
        f"runtime: {ucs_time} seconds."
    )

  
    start_time = time.time()
    result_astar = a_star_search(start)
    astar_time = round(time.time() - start_time, 4)
    print(
        f"A* finished with total cost {result_astar['cost']}, "
        f"state: {result_astar['state']}, "
        f"nodes expanded: {result_astar['expansions']}, "
        f"runtime: {astar_time} seconds."
    )


