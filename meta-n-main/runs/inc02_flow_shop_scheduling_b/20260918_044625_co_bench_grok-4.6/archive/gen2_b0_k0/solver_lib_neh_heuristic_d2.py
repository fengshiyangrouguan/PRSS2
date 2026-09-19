def neh_heuristic(n, m, matrix):
    """NEH heuristic for FFS: produces a good job sequence fast.

    Args:
        n: int — number of jobs.
        m: int — number of machines.
        matrix: list[list[float]] — n x m processing time matrix.

    Returns:
        list[int] — 1-based job indices in the NEH sequence order.
    """
    import itertools
    if n == 0:
        return []
    # Step 1: sort jobs by sum of processing times (descending)
    sums = []
    for i in range(n):
        s = sum(matrix[i])
        sums.append((s, i))
    sums.sort(reverse=True, key=lambda x: x[0])
    seq = [sums[0][1] + 1]  # start with best job (1-based)
    # Step 2: for each remaining job, try all insertion positions and keep best makespan
    for k in range(1, n):
        best_makespan = float('inf')
        best_pos = 0
        for pos in range(k + 1):
            candidate = seq[:pos] + [sums[k][1] + 1] + seq[pos:]
            C = [[0] * m for _ in range(k + 1)]
            for j in range(m):
                for i in range(k + 1):
                    job = candidate[i] - 1
                    if j == 0:
                        C[i][j] = matrix[job][j]
                    else:
                        prev = 0 if i == 0 else C[i - 1][j]
                        C[i][j] = max(prev, C[i][j - 1]) + matrix[job][j]
            makespan = C[k][m - 1]
            if makespan < best_makespan:
                best_makespan = makespan
                best_pos = pos
        seq = seq[:best_pos] + [sums[k][1] + 1] + seq[best_pos:]
    return seq