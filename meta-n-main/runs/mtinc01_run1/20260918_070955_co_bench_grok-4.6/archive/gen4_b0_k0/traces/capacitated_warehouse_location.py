def solve(**kwargs):
    m = kwargs["m"]
    n = kwargs["n"]
    warehouses = kwargs["warehouses"]
    customers = kwargs["customers"]
    # Check total capacity feasibility
    total_capacity = sum(w['capacity'] for w in warehouses)
    total_demand = sum(c['demand'] for c in customers)
    if total_capacity < total_demand:
        return {
            "total_cost": float('inf'),
            "warehouse_open": [0] * m,
            "assignments": [[0.0] * m for _ in range(n)]
        }
    # Greedy assignment: for each customer, assign to lowest cost warehouse with remaining capacity
    assignments = [[0.0] * m for _ in range(n)]
    remaining_capacity = [w['capacity'] for w in warehouses]
    for i in range(n):
        demand = customers[i]['demand']
        costs = customers[i]['costs']
        for _ in range(m):
            min_cost_idx = -1
            min_cost = float('inf')
            for j in range(m):
                if remaining_capacity[j] > 0 and costs[j] < min_cost:
                    min_cost = costs[j]
                    min_cost_idx = j
            if min_cost_idx == -1:
                break
            assign = min(demand, remaining_capacity[min_cost_idx])
            assignments[i][min_cost_idx] = assign
            remaining_capacity[min_cost_idx] -= assign
            demand -= assign
            if demand <= 0:
                break
    # Compute used warehouses
    used = [sum(assignments[i][j] for i in range(n)) for j in range(m)]
    warehouse_open = [1 if u > 0 else 0 for u in used]
    # Compute total_cost
    fixed_cost = sum(warehouses[j]['fixed_cost'] for j in range(m) if used[j] > 0)
    assignment_cost = 0.0
    for i in range(n):
        for j in range(m):
            assignment_cost += assignments[i][j] * customers[i]['costs'][j]
    total_cost = fixed_cost + assignment_cost
    return {
        "total_cost": total_cost,
        "warehouse_open": warehouse_open,
        "assignments": assignments
    }