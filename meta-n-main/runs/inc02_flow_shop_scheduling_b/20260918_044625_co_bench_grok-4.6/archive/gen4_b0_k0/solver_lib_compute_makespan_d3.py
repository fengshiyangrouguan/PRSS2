def compute_makespan(matrix: list[list[float]], perm: list[int]) -> float:
    """Compute makespan for a job permutation on m machines.

    Args:
        matrix: list[list[float]] — matrix[job][machine] = processing time of job on that machine (0-based).
        perm: list[int] — 0-based job order.

    Returns:
        float — makespan value.
    """
    n = len(perm)
    if n == 0:
        return 0.0
    m = len(matrix[0]) if matrix else 0
    C = [[0.0] * m for _ in range(n)]
    for i in range(n):
        job = perm[i]
        for j in range(m):
            if i == 0:
                C[i][j] = matrix[job][j] if j == 0 else C[i][j - 1] + matrix[job][j]
            else:
                C[i][j] = max(C[i - 1][j], C[i][j - 1]) + matrix[job][j]
    return C[n - 1][m - 1]