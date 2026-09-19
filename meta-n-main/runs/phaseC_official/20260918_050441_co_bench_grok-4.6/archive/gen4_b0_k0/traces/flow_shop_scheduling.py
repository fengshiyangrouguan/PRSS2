def solve(**kwargs):
    n = kwargs['n']
    m = kwargs['m']
    matrix = kwargs['matrix']
    if n == 0:
        return {'job_sequence': []}
    INF = 10**9 + 5
    N = 1 << n
    dp = [[INF] * n for _ in range(N)]
    prev = [[-1] * n for _ in range(N)]
    for i in range(n):
        mask = 1 << i
        dp[mask][i] = matrix[i][m - 1]
        prev[mask][i] = -1
    for mask in range(1, N):
        for k in range(n):
            if (mask & (1 << k)) == 0:
                continue
            if mask == (1 << k):
                continue
            min_time = INF
            best_j = -1
            for j in range(n):
                if (mask & (1 << j)) == 0 or j == k:
                    continue
                prev_mask = mask ^ (1 << k)
                if dp[prev_mask][j] == INF:
                    continue
                time = dp[prev_mask][j] + matrix[k][m - 1]
                if time < min_time:
                    min_time = time
                    best_j = j
            dp[mask][k] = min_time
            prev[mask][k] = best_j
    full_mask = (1 << n) - 1
    best_time = INF
    best_k = -1
    for k in range(n):
        if dp[full_mask][k] < best_time:
            best_time = dp[full_mask][k]
            best_k = k
    if best_k == -1:
        return {'job_sequence': list(range(1, n + 1))}
    sequence = []
    mask = full_mask
    cur = best_k
    while cur != -1:
        sequence.append(cur + 1)
        j = prev[mask][cur]
        mask = mask ^ (1 << cur)
        cur = j
    sequence.reverse()
    return {'job_sequence': sequence}