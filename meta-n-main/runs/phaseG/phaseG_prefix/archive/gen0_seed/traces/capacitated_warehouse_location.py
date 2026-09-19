def solve(**kwargs):
    import time
    import math
    import random

    start_time = time.time()
    deadline = start_time + 9.0

    m = kwargs["m"]
    n = kwargs["n"]
    warehouses = kwargs["warehouses"]
    customers = kwargs["customers"]

    capacities = [float(w["capacity"]) for w in warehouses]
    fixed = [float(w["fixed_cost"]) for w in warehouses]
    demands = [float(c["demand"]) for c in customers]
    costs = [list(map(float, c["costs"])) for c in customers]

    total_demand = sum(demands)
    total_capacity = sum(capacities)

    def allocate(open_set):
        """Greedy transportation allocation over a given open set."""
        if time.time() > deadline:
            return None

        cap_sum = sum(capacities[j] for j in open_set)
        if cap_sum + 1e-9 < total_demand:
            return None

        rem_d = demands[:]
        rem_c = [0.0] * m
        for j in open_set:
            rem_c[j] = capacities[j]

        assignments = [[0.0] * m for _ in range(n)]

        pairs = []
        for i in range(n):
            if demands[i] <= 1e-12:
                continue
            ci = costs[i]
            for j in open_set:
                if capacities[j] > 1e-12:
                    pairs.append((ci[j], i, j))

        pairs.sort(key=lambda x: x[0])

        assign_cost = 0.0
        remaining_total = total_demand

        for c, i, j in pairs:
            if remaining_total <= 1e-8:
                break
            if rem_d[i] <= 1e-12 or rem_c[j] <= 1e-12:
                continue
            q = rem_d[i] if rem_d[i] < rem_c[j] else rem_c[j]
            assignments[i][j] = q
            rem_d[i] -= q
            rem_c[j] -= q
            remaining_total -= q
            assign_cost += q * c

        if any(x > 1e-7 for x in rem_d):
            return None

        used = [0] * m
        fixed_cost = 0.0
        for j in range(m):
            used_amount = capacities[j] - rem_c[j]
            if used_amount > 1e-9:
                used[j] = 1
                fixed_cost += fixed[j]

        return fixed_cost + assign_cost, used, assignments

    best = None

    # If total capacity is insufficient, return the least-bad all-open partial allocation.
    # Such an instance is infeasible by definition, but this avoids crashes.
    if total_capacity + 1e-9 < total_demand:
        open_set = [j for j in range(m) if capacities[j] > 1e-12]
        rem_d = demands[:]
        rem_c = capacities[:]
        assignments = [[0.0] * m for _ in range(n)]
        pairs = []
        for i in range(n):
            for j in open_set:
                pairs.append((costs[i][j], i, j))
        pairs.sort(key=lambda x: x[0])
        total_cost = sum(fixed[j] for j in open_set)
        for c, i, j in pairs:
            if rem_d[i] <= 1e-12 or rem_c[j] <= 1e-12:
                continue
            q = rem_d[i] if rem_d[i] < rem_c[j] else rem_c[j]
            assignments[i][j] = q
            rem_d[i] -= q
            rem_c[j] -= q
            total_cost += q * c
        return {
            "total_cost": total_cost,
            "warehouse_open": [1 if j in open_set else 0 for j in range(m)],
            "assignments": assignments,
        }

    def evaluate(open_set):
        nonlocal best
        if not open_set:
            return
        res = allocate(open_set)
        if res is None:
            return
        if best is None or res[0] < best[0]:
            best = res

    all_open = [j for j in range(m) if capacities[j] > 1e-12]
    evaluate(all_open)

    if time.time() > deadline:
        total_cost, used, assignments = best
        return {"total_cost": total_cost, "warehouse_open": used, "assignments": assignments}

    avg_cost = []
    min_cost = []
    for j in range(m):
        s = 0.0
        mn = float("inf")
        for i in range(n):
            v = costs[i][j]
            s += v
            if v < mn:
                mn = v
        avg_cost.append(s / n if n else 0.0)
        min_cost.append(mn if mn < float("inf") else 0.0)

    # Deterministic candidate subsets.
    orderings = []

    orderings.append(sorted(range(m), key=lambda j: fixed[j] / max(capacities[j], 1e-9)))
    orderings.append(sorted(range(m), key=lambda j: avg_cost[j]))
    orderings.append(sorted(range(m), key=lambda j: min_cost[j]))
    orderings.append(sorted(range(m), key=lambda j: fixed[j] / max(capacities[j], 1e-9) + avg_cost[j]))
    orderings.append(sorted(range(m), key=lambda j: fixed[j] / max(capacities[j], 1e-9) + min_cost[j]))

    for alpha in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0):
        orderings.append(
            sorted(
                range(m),
                key=lambda j, a=alpha: avg_cost[j] + a * fixed[j] / max(capacities[j], 1e-9),
            )
        )

    for order in orderings:
        if time.time() > deadline:
            break
        s = []
        cap = 0.0
        for j in order:
            if capacities[j] <= 1e-12:
                continue
            s.append(j)
            cap += capacities[j]
            if cap + 1e-9 >= total_demand:
                break
        evaluate(s)

    # Customer-cheapest driven candidate.
    if time.time() <= deadline:
        chosen = set()
        cap = 0.0
        cust_order = sorted(range(n), key=lambda i: demands[i], reverse=True)
        for i in cust_order:
            ranked = sorted(range(m), key=lambda j: costs[i][j] + fixed[j] / max(capacities[j], 1e-9))
            for j in ranked[:max(1, min(5, m))]:
                if j not in chosen and capacities[j] > 1e-12:
                    chosen.add(j)
                    cap += capacities[j]
                    if cap + 1e-9 >= total_demand:
                        break
            if cap + 1e-9 >= total_demand:
                break
        if cap + 1e-9 < total_demand:
            for j in sorted(range(m), key=lambda x: fixed[x] / max(capacities[x], 1e-9)):
                if j not in chosen and capacities[j] > 1e-12:
                    chosen.add(j)
                    cap += capacities[j]
                    if cap + 1e-9 >= total_demand:
                        break
        evaluate(list(chosen))

    # Local improvement on best used set: try removing warehouses.
    if best is not None and time.time() <= deadline:
        improved = True
        while improved and time.time() <= deadline:
            improved = False
            current_open = [j for j, v in enumerate(best[1]) if v]
            if len(current_open) <= 1:
                break

            # Try removing expensive/low-utilization-looking warehouses first.
            removal_order = sorted(
                current_open,
                key=lambda j: fixed[j] / max(capacities[j], 1e-9),
                reverse=True,
            )

            for j in removal_order:
                if time.time() > deadline:
                    break
                cand = [x for x in current_open if x != j]
                if sum(capacities[x] for x in cand) + 1e-9 < total_demand:
                    continue
                old_cost = best[0]
                evaluate(cand)
                if best[0] + 1e-9 < old_cost:
                    improved = True
                    break

    # Local improvement: try add/remove swaps around current best.
    if best is not None and time.time() <= deadline:
        current = set(j for j, v in enumerate(best[1]) if v)
        closed = [j for j in range(m) if j not in current and capacities[j] > 1e-12]
        closed.sort(key=lambda j: avg_cost[j] + fixed[j] / max(capacities[j], 1e-9))

        tries = 0
        max_tries = 100
        for add_j in closed[:min(len(closed), 20)]:
            if time.time() > deadline or tries >= max_tries:
                break
            base = set(current)
            base.add(add_j)
            evaluate(list(base))
            tries += 1

            current2 = set(j for j, v in enumerate(best[1]) if v)
            for rem_j in list(current2):
                if time.time() > deadline or tries >= max_tries:
                    break
                if rem_j == add_j:
                    continue
                cand = set(current2)
                cand.add(add_j)
                cand.discard(rem_j)
                if sum(capacities[x] for x in cand) + 1e-9 >= total_demand:
                    evaluate(list(cand))
                    tries += 1

    if best is None:
        # Fallback, should rarely happen.
        res = allocate(all_open)
        if res is None:
            assignments = [[0.0] * m for _ in range(n)]
            return {
                "total_cost": 0.0,
                "warehouse_open": [0] * m,
                "assignments": assignments,
            }
        best = res

    total_cost, warehouse_open, assignments = best

    # Recompute objective defensively.
    recomputed = 0.0
    for j in range(m):
        if warehouse_open[j]:
            recomputed += fixed[j]
    for i in range(n):
        for j in range(m):
            if assignments[i][j]:
                recomputed += assignments[i][j] * costs[i][j]

    return {
        "total_cost": recomputed,
        "warehouse_open": warehouse_open,
        "assignments": assignments,
    }