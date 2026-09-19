def compute_makespan(sequence: list, matrix: list) -> float:
    """Computes makespan for a given job permutation on Fm||Cmax using standard DP.

    Args:
        sequence: list of job indices (0-based) in processing order.
        matrix: 2D list of processing times, matrix[j][k] = time of job j on machine k.

    Returns:
        makespan: the Cmax value (makespan) of the sequence.
    """
    n = len(sequence)
    m = len(matrix[0])
    C = [[0.0] * m for _ in range(n + 1)]
    # First row (first job on all machines)
    C[1][0] = matrix[sequence[0]][0]
    for k in range(1, m):
        C[1][k] = C[1][k - 1] + matrix[sequence[0]][k]
    # First column (first machine for remaining jobs)
    for j in range(2, n + 1):
        C[j][0] = C[j - 1][0] + matrix[sequence[j - 1]][0]
        # Remaining cells
        for k in range(1, m):
            C[j][k] = max(C[j - 1][k], C[j][k - 1]) + matrix[sequence[j - 1]][k]
    return C[n][m - 1]