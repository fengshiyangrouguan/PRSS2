def neh_heuristic(n: int, m: int, matrix: list[list[int]]) -> list[int]:
    """NEH heuristic for flow shop makespan minimization.
    Args:
        n: int — number of jobs.
        m: int — number of machines.
        matrix: list[list[int]] — processing times matrix[n_jobs][m_machines].
    Returns:
        list[int] — 0-based job indices in processing order.
    """
    if n <= 1:
        return list(range(n))
    if m == 0:
        return list(range(n))
    # Sort by total processing time descending
    sums = [sum(row) for row in matrix]
    jobs = sorted(range(n), key=lambda j: sums[j], reverse=True)
    seq = jobs[:2]
    if len(seq) == 1:
        pass
    else:
        # Pick better order of first two
        ms1 = makespan(seq, matrix, n, m)
        if makespan(seq[::-1], matrix, n, m) < ms1:
            seq = seq[::-1]
    # Insert remaining jobs
    for j in jobs[2:]:
        best_ms = float('inf')
        best_pos = 0
        for pos in range(len(seq) + 1):
            temp = seq[:pos] + [j] + seq[pos:]
            ms = makespan(temp, matrix, n, m)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq = seq[:best_pos] + [j] + seq[best_pos:]
    return seq