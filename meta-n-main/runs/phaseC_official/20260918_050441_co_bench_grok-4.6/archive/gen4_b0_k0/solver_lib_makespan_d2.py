def makespan(seq: list, matrix: list) -> float:
    """Compute makespan (Cmax) for a given job sequence in a flow shop.

    Args:
        seq: list of job indices (0-based) in the order to process.
        matrix: 2D list where matrix[j][k] is processing time of job j on machine k.

    Returns:
        float: makespan (maximum completion time over all jobs and machines).
    """
    n = len(seq)
    m = len(matrix[0])
    C = [[0.0] * m for _ in range(n)]
    p = [[matrix[j][k] for k in range(m)] for j in seq]
    for j in range(n):
        for k in range(m):
            if j == 0 and k == 0:
                C[j][k] = p[j][k]
            elif j == 0:
                C[j][k] = C[j][k - 1] + p[j][k]
            elif k == 0:
                C[j][k] = C[j - 1][k] + p[j][k]
            else:
                C[j][k] = max(C[j - 1][k], C[j][k - 1]) + p[j][k]
    return C[-1][-1]