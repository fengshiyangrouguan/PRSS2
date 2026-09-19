def solve(**kwargs):
    import time
    import math

    start_time = time.time()
    deadline = start_time + 8.5

    m = kwargs["m"]
    n = kwargs["n"]
    warehouses = kwargs["warehouses"]
    customers = kwargs["customers"]

    capacities = [float(w.get("capacity", 0.0)) for w in warehouses]
    fixed = [float(w.get("fixed_cost", 0.0)) for w in warehouses]
    demands = [float(c.get("demand", 0.0)) for c in customers]
    costs = [list(map(float, c.get("costs", [0.0] * m))) for c in customers]

    total_demand = sum(demands)
    total_capacity = sum(capacities)
    eps = 1e-9

    if n == 0 or total_demand <= eps:
        return {
            "total_cost": 0.0,
            "warehouse_open": [0] * m,
            "assignments": [[0.0] * m for _ in range(n)]
        }

    if m == 0:
        return {
            "total_cost": 0.0,
            "warehouse_open": [],
            "assignments": [[] for _ in range(n)]
        }

    # If infeasible by total capacity, return the best effort with all warehouses open.
    # The evaluator will mark it infeasible, but there is no feasible solution in this case.
    globally_infeasible = total_capacity + 1e-7 < total_demand

    # Precompute each customer's warehouses sorted by assignment cost.
    sorted_wh_by_customer = []
    for i in range(n):
        sorted_wh_by_customer.append(sorted(range(m), key=lambda j: costs[i][j]))

    # Customer order: large demands first, with a mild preference for customers having high cost spread.
    def customer_priority(i):
        row = costs[i]
        mn = min(row) if row else 0.0
        mx = max(row) if row else 0.0
        return -(demands[i] * (1.0 + 0.01 * (mx - mn)))

    customer_order = sorted(range(n), key=customer_priority)

    avg_cost = []
    min_cost = []
    for j in range(m):
        s = 0.0
        mn = float("inf")
        for i in range(n):
            c = costs[i][j]
            s += c * demands[i]
            if c < mn:
                mn = c
        avg_cost.append(s / total_demand if total_demand > eps else 0.0)
        min_cost.append(mn if mn < float("inf") else 0.0)

    def complete_to_capacity(open_set):
        cap = sum(capacities[j] for j in open_set)
        if cap + eps >= total_demand:
            return set(open_set)
        closed = [j for j in range(m) if j not in open_set]
        closed.sort(key=lambda j: (fixed[j] / max(capacities[j], eps), avg_cost[j]))
        res = set(open_set)
        for j in closed:
            if capacities[j] <= eps:
                continue
            res.add(j)
            cap += capacities[j]
            if cap + eps >= total_demand:
                break
        return res

    def evaluate(open_set):
        if time.time() > deadline:
            return None

        open_set = set(j for j in open_set if 0 <= j < m and capacities[j] > eps)
        if not open_set:
            return None

        if sum(capacities[j] for j in open_set) + 1e-7 < total_demand:
            return None

        remaining_cap = capacities[:]
        assignments = [[0.0] * m for _ in range(n)]
        assign_cost = 0.0

        for i in customer_order:
            rem = demands[i]
            if rem <= eps:
                continue
            for j in sorted_wh_by_customer[i]:
                if j not in open_set:
                    continue
                if remaining_cap[j] <= eps:
                    continue
                x = rem if rem < remaining_cap[j] else remaining_cap[j]
                if x > eps:
                    assignments[i][j] = x
                    remaining_cap[j] -= x
                    rem -= x
                    assign_cost += x * costs[i][j]
                    if rem <= 1e-7:
                        break
            if rem > 1e-6:
                return None

        used = [0] * m
        for j in range(m):
            if capacities[j] - remaining_cap[j] > 1e-7:
                used[j] = 1

        total = assign_cost + sum(fixed[j] for j in range(m) if used[j])
        return total, used, assignments

    def build_until_enough(order):
        res = set()
        cap = 0.0
        for j in order:
            if capacities[j] <= eps:
                continue
            res.add(j)
            cap += capacities[j]
            if cap + eps >= total_demand:
                break
        return res

    candidates = []

    # Candidate 1: all warehouses.
    candidates.append(set(j for j in range(m) if capacities[j] > eps))

    # Low fixed-cost-per-capacity.
    candidates.append(build_until_enough(sorted(range(m), key=lambda j: (fixed[j] / max(capacities[j], eps), avg_cost[j]))))

    # Low average assignment cost.
    candidates.append(build_until_enough(sorted(range(m), key=lambda j: (avg_cost[j], fixed[j] / max(capacities[j], eps)))))

    # Low combined scores with several fixed-cost weights.
    avg_fixed_per_cap = sum(fixed) / max(sum(capacities), eps)
    for alpha in (0.25, 0.5, 1.0, 2.0, 4.0):
        order = sorted(
            range(m),
            key=lambda j: avg_cost[j] + alpha * fixed[j] / max(capacities[j], eps)
        )
        candidates.append(build_until_enough(order))

    # Open warehouses that are cheapest for at least one high-demand customer.
    chosen = set()
    for i in sorted(range(n), key=lambda x: -demands[x]):
        if demands[i] <= eps:
            continue
        chosen.add(sorted_wh_by_customer[i][0])
        if sum(capacities[j] for j in chosen) + eps >= total_demand:
            break
    candidates.append(complete_to_capacity(chosen))

    # Cheapest few choices per customer, useful when demand is geographically clustered.
    for k in (1, 2, 3):
        chosen = set()
        for i in range(n):
            if demands[i] <= eps:
                continue
            for j in sorted_wh_by_customer[i][:min(k, m)]:
                chosen.add(j)
        candidates.append(complete_to_capacity(chosen))

    best = None

    for cand in candidates:
        if time.time() > deadline:
            break
        res = evaluate(cand)
        if res is not None and (best is None or res[0] < best[0]):
            best = res

    # Local improvement: try removing open warehouses from the current best solution.
    if best is not None and not globally_infeasible:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_open = {j for j, v in enumerate(best[1]) if v}
            removal_order = sorted(
                current_open,
                key=lambda j: fixed[j] - 0.01 * capacities[j],
                reverse=True
            )
            for j in removal_order:
                if time.time() > deadline:
                    break
                trial = set(current_open)
                trial.remove(j)
                if sum(capacities[x] for x in trial) + 1e-7 < total_demand:
                    continue
                res = evaluate(trial)
                if res is not None and res[0] + 1e-7 < best[0]:
                    best = res
                    improved = True
                    break

    if best is None:
        # Last-resort deterministic construction with all warehouses.
        open_set = set(j for j in range(m) if capacities[j] > eps)
        remaining_cap = capacities[:]
        assignments = [[0.0] * m for _ in range(n)]
        assign_cost = 0.0
        for i in range(n):
            rem = demands[i]
            for j in sorted_wh_by_customer[i]:
                if j not in open_set:
                    continue
                x = min(rem, remaining_cap[j])
                if x > eps:
                    assignments[i][j] = x
                    remaining_cap[j] -= x
                    rem -= x
                    assign_cost += x * costs[i][j]
                if rem <= 1e-7:
                    break
        used = [1 if capacities[j] - remaining_cap[j] > 1e-7 else 0 for j in range(m)]
        total = assign_cost + sum(fixed[j] for j in range(m) if used[j])
        best = (total, used, assignments)

    total_cost, warehouse_open, assignments = best

    # Tiny numerical cleanup: avoid negative zeros and very small values.
    for i in range(n):
        for j in range(m):
            if abs(assignments[i][j]) < 1e-10:
                assignments[i][j] = 0.0

    return {
        "total_cost": float(total_cost),
        "warehouse_open": [int(x) for x in warehouse_open],
        "assignments": assignments
    }