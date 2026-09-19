def neh(matrix: list[list[int]], n: int, m: int) -> list[int]:
    """NEH heuristic for flow-shop scheduling.
    Args:
        matrix: list[list[int]] — processing times; matrix[job][machine]
        n: int — number of jobs
        m: int — number of machines
    Returns:
        list[int] — 0-based job order (1-based in final output)
    """
    if n == 0:
        return []
    if m == 0:
        return list(range(n))
    # Compute total processing time per job
    totals = [sum(row) for row in matrix]
    # Initial sort by total time descending
    jobs = sorted(range(n), key=lambda j: totals[j], reverse=True)
    # Build sequence by inserting one by one
    seq = [jobs[0]]
    for j in range(1, n):
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(seq) + 1):
            # insert at pos
            temp = seq[:pos] + [jobs[j]] + seq[pos:]
            ms = makespan(temp, matrix, len(temp), m)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq = seq[:best_pos] + [jobs[j]] + seq[best_pos:]
    return seq