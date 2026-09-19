def neh_heuristic(n: int, m: int, matrix: list) -> list:
    """NEH insertion heuristic for m-machine flow shop (strong practical approximation).

    Args:
        n: number of jobs
        m: number of machines
        matrix: n x m list of processing times (matrix[i][j] = time of job i on machine j)

    Returns:
        list of job indices (0-based) in the order to process them.
    """
    if n == 0:
        return []
    # Longest-processing-time initial order
    p = [sum(row) for row in matrix]
    jobs = list(range(n))
    jobs.sort(key=p.__getitem__, reverse=True)
    if n == 1:
        return [jobs[0]]
    # Start with first two jobs
    seq = [jobs[0], jobs[1]]
    for k in range(2, n):
        best_makespan = float('inf')
        best_pos = 0
        job = jobs[k]
        for pos in range(len(seq) + 1):
            temp_seq = seq[:pos] + [job] + seq[pos:]
            # Correct full 2D makespan evaluation (O(n*m) per trial)
            C = [[0.0] * m for _ in range(n)]
            for i in range(len(temp_seq)):
                j = temp_seq[i]
                for mm in range(m):
                    if i == 0 and mm == 0:
                        C[i][mm] = matrix[j][mm]
                    elif mm == 0:
                        C[i][mm] = C[i - 1][mm] + matrix[j][mm]
                    elif i == 0:
                        C[i][mm] = C[i][mm - 1] + matrix[j][mm]
                    else:
                        C[i][mm] = max(C[i - 1][mm], C[i][mm - 1]) + matrix[j][mm]
            ms = C[-1][-1]
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        seq = seq[:best_pos] + [job] + seq[best_pos:]
    return seq