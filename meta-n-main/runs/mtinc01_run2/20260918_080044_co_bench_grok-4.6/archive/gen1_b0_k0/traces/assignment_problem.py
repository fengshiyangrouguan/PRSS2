def solve(**kwargs):
    n = kwargs['n']
    cost_matrix = kwargs['cost_matrix']
    from itertools import permutations
    min_cost = float('inf')
    best_perm = None
    for p in permutations(range(n)):
        total = sum(cost_matrix[i][p[i]] for i in range(n))
        if total < min_cost:
            min_cost = total
            best_perm = p
    assignment = [(i + 1, j + 1) for i, j in enumerate(best_perm)]
    return {"total_cost": min_cost, "assignment": assignment}