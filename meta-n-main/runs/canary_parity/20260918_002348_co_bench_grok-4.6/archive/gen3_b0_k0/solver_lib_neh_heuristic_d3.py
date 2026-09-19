def neh_heuristic(n: int, m: int, matrix: list[list[int]]) -> list[int]:
    """NEH heuristic for flow-shop scheduling (minimizes makespan).
    Sorts jobs by total processing time then inserts each subsequent job in the best position.
    Args:
        n: int — number of jobs (0-based indexing assumed)
        m: int — number of machines
        matrix: list[list[int]] — n x m processing time matrix; matrix[job][machine]
    Returns:
        list[int] — job sequence (0-based indices) produced by NEH
    """
    if n <= 0:
        return []
    # totals for sorting
    totals = [sum(row) for row in matrix]
    jobs = sorted(range(n), key=lambda j: totals[j], reverse=True)
    if n == 1:
        return [jobs[0]]
    # try both orders on first two jobs
    best_seq = None
    best_makespan = float('inf')
    for perm in [[jobs[0], jobs[1]], [jobs[1], jobs[0]]]:
        seq = perm[:]
        for i in range(2, n):
            job = jobs[i]
            best_insert = None
            best_m = float('inf')
            for pos in range(len(seq) + 1):
                temp_seq = seq[:pos] + [job] + seq[pos:]
                mspan = _compute_makespan(temp_seq, matrix, m)
                if mspan < best_m:
                    best_m = mspan
                    best_insert = pos
            seq = seq[:best_insert] + [job] + seq[best_insert:]
        mspan = _compute_makespan(seq, matrix, m)
        if mspan < best_makespan:
            best_makespan = mspan
            best_seq = seq
    return best_seq