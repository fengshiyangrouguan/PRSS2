def solve(**kwargs):
    import time

    start_time = time.time()
    deadline = start_time + 8.7

    m = kwargs["m"]
    n = kwargs["n"]
    warehouses = kwargs["warehouses"]
    customers = kwargs["customers"]

    capacities = [float(w.get("capacity", 0.0)) for w in warehouses]
    fixed = [float(w.get("fixed_cost", 0.0)) for w in warehouses]
    demands = [float(c.get("demand", 0.0)) for c in customers]
    costs = [c.get("costs", [0.0] * m) for c in customers]

    total_demand = sum(demands)
    total_capacity = sum(capacities)

    if n == 0 or total_demand <= 1e-12:
        return {
            "total_cost": 0.0,
            "warehouse_open": [0] * m,
            "assignments": [[0.0] * m for _ in range(n)]
        }

    if m == 0 or total_capacity + 1e-9 < total_demand:
        return {
            "total_cost": 0.0,
            "warehouse_open": [0] * m,
            "assignments": [[0.0] * m for _ in range(n)]
        }

    arcs = []
    for i in range(n):
        ci = costs[i]
        for j in range(m):
            arcs.append((float(ci[j]), i, j))
    arcs.sort(key=lambda x: x[0])

    def build_solution(open_mask):
        rem_d = demands[:]
        rem_c = capacities[:]
        assignments = [[0.0] * m for _ in range(n)]
        variable_cost = 0.0
        unsatisfied = total_demand

        for c, i, j in arcs:
            if not open_mask[j]:
                continue
            if rem_d[i] <= 1e-10 or rem_c[j] <= 1e-10:
                continue
            q = rem_d[i] if rem_d[i] < rem_c[j] else rem_c[j]
            if q > 0.0:
                assignments[i][j] = q
                rem_d[i] -= q
                rem_c[j] -= q
                unsatisfied -= q
                variable_cost += q * c
                if unsatisfied <= 1e-8:
                    break

        if unsatisfied > 1e-6:
            return None

        used = [0] * m
        for j in range(m):
            if capacities[j] - rem_c[j] > 1e-8:
                used[j] = 1

        fixed_cost = sum(fixed[j] for j in range(m) if used[j])
        return variable_cost + fixed_cost, used, assignments

    all_open = [capacities[j] > 1e-12 for j in range(m)]
    best = build_solution(all_open)

    if best is None:
        return {
            "total_cost": 0.0,
            "warehouse_open": [0] * m,
            "assignments": [[0.0] * m for _ in range(n)]
        }

    best_cost, best_open, best_assign = best

    def mask_capacity(mask):
        s = 0.0
        for jj in range(m):
            if mask[jj]:
                s += capacities[jj]
        return s

    current_mask = [bool(x) for x in best_open]
    current_cap = mask_capacity(current_mask)

    candidates = [j for j in range(m) if current_mask[j]]
    candidates.sort(key=lambda j: fixed[j], reverse=True)

    max_trials = min(len(candidates), 80 if m * n > 20000 else 200)

    improved = True
    passes = 0
    while improved and passes < 3 and time.time() < deadline:
        improved = False
        passes += 1
        candidates = [j for j in range(m) if current_mask[j]]
        candidates.sort(key=lambda j: fixed[j], reverse=True)

        tried = 0
        for j in candidates:
            if time.time() >= deadline or tried >= max_trials:
                break
            tried += 1

            if not current_mask[j]:
                continue
            if current_cap - capacities[j] + 1e-9 < total_demand:
                continue

            trial_mask = current_mask[:]
            trial_mask[j] = False
            sol = build_solution(trial_mask)
            if sol is None:
                continue

            cost, op, ass = sol
            if cost + 1e-7 < best_cost:
                best_cost, best_open, best_assign = cost, op, ass
                current_mask = [bool(x) for x in op]
                current_cap = mask_capacity(current_mask)
                improved = True
                break

    # Try a few constructive alternatives biased by low fixed cost and low average assignment cost.
    if time.time() < deadline:
        avg_cost = []
        sample_customers = range(n)
        for j in range(m):
            weighted = 0.0
            dem = 0.0
            for i in sample_customers:
                weighted += demands[i] * float(costs[i][j])
                dem += demands[i]
            avg = weighted / dem if dem > 1e-12 else 0.0
            avg_cost.append(avg)

        orderings = []
        orderings.append(sorted(range(m), key=lambda j: fixed[j] / (capacities[j] + 1e-9)))
        orderings.append(sorted(range(m), key=lambda j: avg_cost[j]))
        orderings.append(sorted(range(m), key=lambda j: avg_cost[j] + fixed[j] / (capacities[j] + 1e-9)))

        for order in orderings:
            if time.time() >= deadline:
                break
            mask = [False] * m
            cap = 0.0
            for j in order:
                mask[j] = True
                cap += capacities[j]
                if cap + 1e-9 >= total_demand:
                    break
            if cap + 1e-9 < total_demand:
                continue
            sol = build_solution(mask)
            if sol is not None and sol[0] + 1e-7 < best_cost:
                best_cost, best_open, best_assign = sol

    # Clean tiny numerical residue and recompute exact reported cost.
    warehouse_load = [0.0] * m
    for i in range(n):
        row_sum = sum(best_assign[i])
        diff = demands[i] - row_sum
        if abs(diff) > 1e-7:
            for j in range(m):
                if best_assign[i][j] > 1e-9 and warehouse_load[j] + diff <= capacities[j] + 1e-7:
                    best_assign[i][j] += diff
                    break
        for j in range(m):
            if best_assign[i][j] < 1e-10:
                best_assign[i][j] = 0.0
            warehouse_load[j] += best_assign[i][j]

    final_open = [1 if warehouse_load[j] > 1e-8 else 0 for j in range(m)]
    final_cost = sum(fixed[j] for j in range(m) if final_open[j])
    for i in range(n):
        for j in range(m):
            if best_assign[i][j] != 0.0:
                final_cost += best_assign[i][j] * float(costs[i][j])

    return {
        "total_cost": float(final_cost),
        "warehouse_open": final_open,
        "assignments": best_assign
    }