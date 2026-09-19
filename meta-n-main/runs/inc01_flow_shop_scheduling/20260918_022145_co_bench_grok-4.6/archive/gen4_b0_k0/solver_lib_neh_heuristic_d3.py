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
    # Start with first two jobs (standard NEH variant)
    if n == 1:
        return [jobs[0]]
    current = [jobs[0], jobs[1]]
    for k in range(2, n):
        best_makespan = float('inf')
        best_pos = 0
        for pos in range(len(current) + 1):
            temp_seq = current[:pos] + [jobs[k]] + current[pos:]
            # Correct full 2D makespan evaluation (no incremental reset bug)
            C = [[0.0] * m for _ in range(len(temp_seq))]
            for i in range(len(temp_seq)):
                job = temp_seq[i]
                for j in range(m):
                    if i == 0 and j == 0:
                        C[i][j] = matrix[job][j]
                    elif i == 0:
                        C[i][j] = C[i][j-1] + matrix[job][j]
                    elif j == 0:
                        C[i][j] = C[i-1][j] + matrix[job][j]
                    else:
                        C[i][j] = max(C[i-1][j], C[i][j-1]) + matrix[job][j]
            ms = C[-1][-1]
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        current = current[:best_pos] + [jobs[k]] + current[best_pos:]
    return current