from assignment import INF, Problem, default_instance, random_instance, search, hungarian_cost


def instances():
    return [default_instance()] + [random_instance(n, n + 3, s) for n in (4, 6, 8) for s in range(5)]


def test_all_methods_match_hungarian():
    for p in instances():
        base = hungarian_cost(p, p.rides, p.employees)
        for h in (p.h_zero, p.h_favorite, p.h_hungarian):
            res = search(p, h)
            if base == INF:
                assert res is None  # infeasible instance handled cleanly
            else:
                assert res["cost"] == base  # optimal


def test_heuristics_admissible():
    for p in instances():
        base = hungarian_cost(p, p.rides, p.employees)
        if base == INF:
            continue
        s = p.initial_state()
        assert p.h_favorite(s) <= base  # never overestimates
        assert p.h_hungarian(s) <= base


def test_infeasible():
    p = Problem(["A", "B"], ["E1"], {"E1": ["A", "B"]}, {"E1": "A"})
    assert search(p, p.h_zero) is None