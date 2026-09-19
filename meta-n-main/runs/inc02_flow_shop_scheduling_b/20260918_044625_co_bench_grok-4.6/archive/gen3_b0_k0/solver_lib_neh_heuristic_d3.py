def neh_heuristic(n: int, m: int, matrix: list[list[float]]) -> list[int]:
    """NEH heuristic for FFS: produces a good job sequence fast.

    Args:
        n: int — number of jobs.
        m: int — number of machines.
        matrix: list[list[float]] — n x m processing time matrix.

    Returns:
        list[int] — 1-based job indices in the NEH sequence order.
    """
    if n == 0:
        return []
    # Step 1: sort jobs by sum of processing times (descending)
    sums = [(sum(matrix[i]), i) for i in range(n)]
    sums.sort(reverse=True)
    seq = [sums[0][1] + 1]
    # Step 2: for each remaining job, try all insertion positions and keep best makespan
    for k in range(1, n):
        best_makespan = float('inf')
        best_pos = 0
        for pos in range(k + 1):
            candidate = seq[:pos] + [sums[k][1] + 1] + seq[pos:]
            # Correct C-matrix computation (handles machine 0 chain)
            C = [[0] * m for _ in range(k + 1)]
            for i in range(k + 1):
                job = candidate[i] - 1
                for j in range(m):
                    if i == 0 and j == 0:
                        C[i][j] = matrix[job][j]
                    elif i == 0:
                        C[i][j] = C[i][j - 1] + matrix[job][j]
                    elif j == 0:
                        C[i][j] = C[i - 1][j] + matrix[job][j]
                    else:
                        C[i][j] = max(C[i - 1][j], C[i][j - 1]) + matrix[job][j]
            makespan = C[k][m - 1]
            if makespan < best_makespan:
                best_makespan = makespan
                best_pos = pos
        seq = seq[:best_pos] + [sums[k][1] + 1] + seq[best_pos:]
    return seq