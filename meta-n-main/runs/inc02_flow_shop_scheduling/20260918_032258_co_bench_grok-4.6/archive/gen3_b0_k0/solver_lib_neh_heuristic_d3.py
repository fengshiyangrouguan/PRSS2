def neh_heuristic(n: int, m: int, matrix: list[list[int]]) -> list[int]:
    """NEH heuristic for flow-shop scheduling.
    Args:
        n: int — number of jobs.
        m: int — number of machines.
        matrix: list[list[int]] — matrix[i][j] = processing time of job i on machine j (0-based).
    Returns:
        list[int] — 0-based job indices in NEH order.
    """
    if n == 0:
        return []
    if n == 1:
        return [0]
    totals = [sum(row) for row in matrix]
    jobs = sorted(range(n), key=lambda i: totals[i], reverse=True)
    best_seq = [jobs[0], jobs[1]]
    for i in range(2, n):
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(best_seq) + 1):
            temp_seq = best_seq[:pos] + [jobs[i]] + best_seq[pos:]
            ms = makespan(temp_seq, matrix, n, m)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        best_seq = best_seq[:best_pos] + [jobs[i]] + best_seq[best_pos:]
    return best_seq