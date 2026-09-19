def makespan(seq, matrix, n, m):
    """Compute makespan for flow-shop scheduling given job sequence.
    Args:
        seq: list[int] — job indices (0-based) in the order to process.
        matrix: list[list[int]] — processing times; matrix[i][j] is job i on machine j.
        n: int — number of jobs.
        m: int — number of machines.
    Returns:
        int — makespan C_max.
    """
    if n == 0:
        return 0
    if m == 0:
        return sum(sum(row) for row in matrix)  # all work done
    C = [[0] * m for _ in range(n)]
    for j in range(m):
        if j == 0:
            C[0][j] = matrix[seq[0]][j]
        else:
            C[0][j] = C[0][j - 1] + matrix[seq[0]][j]
    for i in range(1, n):
        C[i][0] = C[i - 1][0] + matrix[seq[i]][0]
        for j in range(1, m):
            C[i][j] = max(C[i - 1][j], C[i][j - 1]) + matrix[seq[i]][j]
    return C[n - 1][m - 1]