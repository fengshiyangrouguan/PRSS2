def solve(**kwargs):
    n = kwargs['n']
    m = kwargs['m']
    matrix = kwargs['matrix']
    if n == 0:
        return {'job_sequence': []}
    from collections import defaultdict
    dp = defaultdict(dict)
    pred = defaultdict(dict)
    INF = float('inf')
    for j in range(n):
        S = frozenset([j])
        comp = [matrix[j][mm] for mm in range(m)]
        dp[S][j] = tuple(comp)
        pred[S][j] = (None, None)
    for _ in range(n - 1):
        for S_tup in list(dp.keys()):
            S = S_tup
            for j in list(dp[S].keys()):
                comp = dp[S][j]
                for k in range(n):
                    if k in S:
                        continue
                    new_comp = [0] * m
                    start = comp[0]
                    new_comp[0] = start + matrix[k][0]
                    for mm in range(1, m):
                        start = max(comp[mm], new_comp[mm - 1])
                        new_comp[mm] = start + matrix[k][mm]
                    new_S = S | {k}
                    if new_S not in dp or new_comp < dp[new_S].get(k, (INF,) * m):
                        dp[new_S][k] = tuple(new_comp)
                        pred[new_S][k] = (S, j)
    min_makespan = INF
    best_S = None
    best_j = None
    for S in dp:
        for j in dp[S]:
            if dp[S][j][m - 1] < min_makespan:
                min_makespan = dp[S][j][m - 1]
                best_S = S
                best_j = j
    sequence = []
    curr_S = best_S
    curr_j = best_j
    for _ in range(n):
        sequence.append(curr_j + 1)
        prev_S, prev_j = pred[curr_S][curr_j]
        curr_S = prev_S
        curr_j = prev_j
    sequence.reverse()
    return {'job_sequence': sequence}