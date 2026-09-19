def compute_makespan(seq: list, matrix: list, n: int, m: int) -> int:
    """Compute makespan for a given job sequence in a flow-shop scheduling problem using the standard recurrence.

    Args:
        seq: list of job indices (0..n-1) in the order to process.
        matrix: 2D list where matrix[i][j] is processing time of job i on machine j.
        n: number of jobs.
        m: number of machines.

    Returns:
        makespan: integer makespan (last machine last job completion time).
    """
    C = [[0] * m for _ in range(n)]
    for j in range(m):
        C[0][j] = matrix[seq[0]][j] if j == 0 else C[0][j - 1] + matrix[seq[0]][j]
    for i in range(1, n):
        C[i][0] = C[i - 1][0] + matrix[seq[i]][0]
    for i in range(1, n):
        for j in range(1, m):
            C[i][j] = matrix[seq[i]][j] + max(C[i - 1][j], C[i][j - 1])
    return C[n - 1][m - 1]