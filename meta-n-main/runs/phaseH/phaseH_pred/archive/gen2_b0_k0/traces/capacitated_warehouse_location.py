def solve(**kwargs):
    import time
    import math
    import random

    start_time = time.time()
    deadline = start_time + 9.2

    m = int(kwargs["m"])
    n = int(kwargs["n"])
    warehouses = kwargs["warehouses"]
    customers = kwargs["customers"]

    caps = [float(w.get("capacity", 0.0)) for w in warehouses]
    fixed = [float(w.get("fixed_cost", 0.0)) for w in warehouses]
    demands = [float(c.get("demand", 0.0)) for c in customers]
    costs = [list(map(float, c.get("costs", [0.0] * m))) for c in customers]

    total_demand = sum(demands)
    total_capacity = sum(caps)

    assignments_zero = [[0.0] * m for _ in range(n)]

    if n == 0 or total_demand <= 1e-12:
        return {
            "total_cost": 0.0,
            "warehouse_open": [0] * m,
            "assignments": assignments_zero,
        }

    if m == 0 or total_capacity + 1e-9 < total_demand:
        return {
            "total_cost": 0.0,
            "warehouse_open": [0] * m,
            "assignments": assignments_zero,
        }

    # Precompute warehouse order by cost for each customer.
    sorted_wh_by_customer = []
    customer_regret = []
    for i in range(n):
        order = sorted(range(m), key=lambda j: costs[i][j])
        sorted_wh_by_customer.append(order)
        if m >= 2:
            customer_regret.append(costs[i][order[1]] - costs[i][order[0]])
        elif m == 1:
            customer_regret.append(0.0)

    customer_order = sorted(range(n), key=lambda i: (-customer_regret[i], -demands[i]))

    avg_cost = []
    for j in range(m):
        s = 0.0
        for i in range(n):
            s += costs[i][j] * demands[i]
        avg_cost.append(s / total_demand if total_demand > 0 else 0.0)

    def evaluate(open_bool):
        rem = caps[:]
        opened = [1 if open_bool[j] and caps[j] > 1e-12 else 0 for j in range(m)]
        if sum(caps[j] for j in range(m) if opened[j]) + 1e-8 < total_demand:
            return None

        ass = [[0.0] * m for _ in range(n)]
        variable_cost = 0.0

        for i in customer_order:
            need = demands[i]
            if need <= 1e-12:
                continue
            for j in sorted_wh_by_customer[i]:
                if not opened[j]:
                    continue
                if rem[j] <= 1e-12:
                    continue
                x = need if need < rem[j] else rem[j]
                if x > 0.0:
                    ass[i][j] = x
                    rem[j] -= x
                    need -= x
                    variable_cost += x * costs[i][j]
                    if need <= 1e-8:
                        break
            if need > 1e-7:
                return None

        # Close warehouses that were selected but ended up unused.
        used = [0.0] * m
        for i in range(n):
            row = ass[i]
            for j in range(m):
                used[j] += row[j]
        for j in range(m):
            if used[j] <= 1e-9:
                opened[j] = 0

        total_cost = variable_cost + sum(fixed[j] for j in range(m) if opened[j])
        return total_cost, opened, ass

    best = None

    def consider(open_bool):
        nonlocal best
        if time.time() > deadline:
            return
        res = evaluate(open_bool)
        if res is not None and (best is None or res[0] < best[0]):
            best = res

    # Candidate 1: all warehouses.
    consider([1] * m)

    # Candidate prefixes under several scoring rules.
    min_fixed = min(fixed) if fixed else 0.0
    max_fixed = max(fixed) if fixed else 1.0
    min_avg = min(avg_cost) if avg_cost else 0.0
    max_avg = max(avg_cost) if avg_cost else 1.0
    fixed_scale = max(max_fixed - min_fixed, 1.0)
    avg_scale = max(max_avg - min_avg, 1.0)

    scoring_rules = []

    scoring_rules.append(lambda j: fixed[j] / (caps[j] + 1e-9))
    scoring_rules.append(lambda j: avg_cost[j])
    scoring_rules.append(lambda j: avg_cost[j] + fixed[j] / (caps[j] + 1e-9))
    scoring_rules.append(lambda j: (fixed[j] - min_fixed) / fixed_scale + (avg_cost[j] - min_avg) / avg_scale)
    scoring_rules.append(lambda j: 0.35 * (fixed[j] - min_fixed) / fixed_scale + (avg_cost[j] - min_avg) / avg_scale)
    scoring_rules.append(lambda j: (fixed[j] - min_fixed) / fixed_scale + 0.35 * (avg_cost[j] - min_avg) / avg_scale)
    scoring_rules.append(lambda j: fixed[j])
    scoring_rules.append(lambda j: -caps[j])

    for rule in scoring_rules:
        if time.time() > deadline:
            break
        order = sorted(range(m), key=rule)
        open_bool = [0] * m
        cap_sum = 0.0
        for j in order:
            open_bool[j] = 1
            cap_sum += caps[j]
            if cap_sum + 1e-9 >= total_demand:
                break
        consider(open_bool)

        # Also try adding a few cheapest-transport warehouses to this prefix.
        if time.time() <= deadline:
            ext = open_bool[:]
            for j in sorted(range(m), key=lambda x: avg_cost[x])[:max(1, min(m, 5))]:
                ext[j] = 1
            consider(ext)

    # Customer-driven candidate: open cheapest warehouses for largest demands until enough capacity.
    if time.time() <= deadline:
        open_bool = [0] * m
        cap_sum = 0.0
        cust_by_demand = sorted(range(n), key=lambda i: -demands[i])
        for i in cust_by_demand:
            for j in sorted_wh_by_customer[i][:min(m, 3)]:
                if not open_bool[j]:
                    open_bool[j] = 1
                    cap_sum += caps[j]
                    if cap_sum + 1e-9 >= total_demand:
                        break
            if cap_sum + 1e-9 >= total_demand:
                break
        if cap_sum + 1e-9 < total_demand:
            for j in sorted(range(m), key=lambda x: fixed[x] / (caps[x] + 1e-9)):
                if not open_bool[j]:
                    open_bool[j] = 1
                    cap_sum += caps[j]
                    if cap_sum + 1e-9 >= total_demand:
                        break
        consider(open_bool)

    # Local search: close costly warehouses if it improves the evaluated solution.
    improved = True
    while improved and time.time() <= deadline:
        improved = False
        if best is None:
            break
        current_open = best[1][:]
        close_order = sorted(
            [j for j in range(m) if current_open[j]],
            key=lambda j: (-fixed[j], avg_cost[j])
        )
        for j in close_order:
            if time.time() > deadline:
                break
            trial = current_open[:]
            trial[j] = 0
            if sum(caps[k] for k in range(m) if trial[k]) + 1e-9 < total_demand:
                continue
            before = best[0]
            consider(trial)
            if best is not None and best[0] + 1e-9 < before:
                current_open = best[1][:]
                improved = True
                break

    # Small add/drop refinement.
    if best is not None and time.time() <= deadline:
        base_open = best[1][:]
        closed = [j for j in range(m) if not base_open[j]]
        for add_j in sorted(closed, key=lambda j: avg_cost[j] + fixed[j] / (caps[j] + 1e-9))[:min(12, len(closed))]:
            if time.time() > deadline:
                break
            trial = base_open[:]
            trial[add_j] = 1
            consider(trial)

    if best is None:
        # Last-resort feasible construction with all warehouses.
        res = evaluate([1] * m)
        if res is None:
            return {
                "total_cost": 0.0,
                "warehouse_open": [0] * m,
                "assignments": assignments_zero,
            }
        best = res

    total_cost, warehouse_open, assignments = best

    # Numerical cleanup: remove tiny negatives/noise and recompute cost consistently.
    for i in range(n):
        for j in range(m):
            if abs(assignments[i][j]) < 1e-10:
                assignments[i][j] = 0.0

    used = [0.0] * m
    variable_cost = 0.0
    for i in range(n):
        for j in range(m):
            x = assignments[i][j]
            used[j] += x
            variable_cost += x * costs[i][j]

    warehouse_open = [1 if used[j] > 1e-9 else 0 for j in range(m)]
    total_cost = variable_cost + sum(fixed[j] for j in range(m) if warehouse_open[j])

    return {
        "total_cost": float(total_cost),
        "warehouse_open": warehouse_open,
        "assignments": assignments,
    }