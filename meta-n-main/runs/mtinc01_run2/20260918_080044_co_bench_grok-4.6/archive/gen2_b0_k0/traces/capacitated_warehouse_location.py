def solve(**kwargs):
    m = kwargs["m"]
    n = kwargs["n"]
    warehouses = kwargs["warehouses"]
    customers = kwargs["customers"]
    total_cap = sum(w['capacity'] for w in warehouses)
    total_dem = sum(c['demand'] for c in customers)
    if total_cap < total_dem:
        return {
            "total_cost": 0.0,
            "warehouse_open": [0] * m,
            "assignments": [[0.0] * m for _ in range(n)]
        }
    capacities = [w['capacity'] for w in warehouses]
    assignments = [[0.0] * m for _ in range(n)]
    for i in range(n):
        dem = customers[i]['demand']
        costs = customers[i]['costs']
        sorted_indices = sorted(range(m), key=lambda j: costs[j])
        for j in sorted_indices:
            if capacities[j] > 0:
                assign = min(dem, capacities[j])
                assignments[i][j] = assign
                capacities[j] -= assign
                dem -= assign
                if dem == 0:
                    break
    open_list = [0] * m
    total_assign_cost = 0.0
    for i in range(n):
        for j in range(m):
            if assignments[i][j] > 0:
                open_list[j] = 1
                total_assign_cost += assignments[i][j] * customers[i]['costs'][j]
    total_cost = sum(warehouses[j]['fixed_cost'] for j in range(m) if open_list[j]) + total_assign_cost
    return {
        "total_cost": total_cost,
        "warehouse_open": open_list,
        "assignments": assignments
    }