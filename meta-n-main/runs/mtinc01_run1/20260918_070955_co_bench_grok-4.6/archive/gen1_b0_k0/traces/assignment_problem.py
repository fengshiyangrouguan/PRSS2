def solve(**kwargs):
    import math
    n = kwargs['n']
    cost_matrix = kwargs['cost_matrix']
    INF = float('inf')
    N = 1 << n
    pop = [0] * N
    for i in range(1, N):
        pop[i] = pop[i >> 1] + (i & 1)
    dp = [INF] * N
    dp[0] = 0
    for mask in range(N):
        k = pop[mask]
        if k == n:
            continue
        for j in range(n):
            if (mask & (1 << j)) == 0:
                new_mask = mask | (1 << j)
                dp[new_mask] = min(dp[new_mask], dp[mask] + cost_matrix[k][j])
    min_cost = dp[N - 1]
    mask = N - 1
    assignment = []
    for k in range(n - 1, -1, -1):
        for j in range(n):
            if (mask & (1 << j)) == 0:
                continue
            new_mask = mask ^ (1 << j)
            if dp[new_mask] + cost_matrix[k][j] == dp[mask]:
                assignment.append((k + 1, j + 1))
                mask = new_mask
                break
    assignment.reverse()
    return {"total_cost": min_cost, "assignment": assignment}