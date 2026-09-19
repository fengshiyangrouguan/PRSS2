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
    C = [[0.0] * m for _ in range(n)]
    # First job on all machines
    C[0][0] = matrix[sequence[0]][0]
    for k in range(1, m):
        C[0][k] = C[0][k - 1] + matrix[sequence[0]][k]
    # First machine for remaining jobs
    for i in range(1, n):
        C[i][0] = C[i - 1][0] + matrix[sequence[i]][0]
    # Remaining machines
    for k in range(1, m):
        for i in range(1, n):
            C[i][k] = max(C[i - 1][k], C[i][k - 1]) + matrix[sequence[i]][k]
    return C[n - 1][m - 1]