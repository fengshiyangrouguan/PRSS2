def neh_heuristic(matrix: list[list[int]]) -> list[int]:
    """NEH heuristic for Fm||Cmax (best-known approximation ratio ~0.794).

    Args:
        matrix: 2-D list; matrix[i][k] = processing time of job i on machine k (0-based).

    Returns:
        list[int]: 0-based job indices in the NEH order.
    """
    n = len(matrix)
    # 1. Sort jobs by total processing time (descending)
    jobs = sorted(range(n), key=lambda j: sum(matrix[j]), reverse=True)
    # 2. Insert one-by-one
    seq = [jobs[0]]
    for i in range(1, n):
        best_pos = 0
        best_makespan = float('inf')
        for pos in range(len(seq) + 1):
            candidate = seq[:pos] + [jobs[i]] + seq[pos:]
            makespan = compute_makespan(matrix, candidate)
            if makespan < best_makespan:
                best_makespan = makespan
                best_pos = pos
        seq = seq[:best_pos] + [jobs[i]] + seq[best_pos:]
    return seq