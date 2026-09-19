def makespan(seq: list, matrix: list) -> float:
    """Compute makespan for a given job sequence on the flow-shop matrix.

    Args:
        seq: list of 0-based job indices in the order to process.
        matrix: 2-D list where matrix[j][k] is processing time of job j on machine k.

    Returns:
        Makespan (float) — the completion time of the last job on the last machine.
    """
    n = len(seq)
    m = len(matrix[0])
    C = [[0.0] * m for _ in range(n)]
    for j in range(n):
        for k in range(m):
            if j == 0 and k == 0:
                C[j][k] = matrix[seq[j]][k]
            elif j == 0:
                C[j][k] = C[j][k - 1] + matrix[seq[j]][k]
            elif k == 0:
                C[j][k] = C[j - 1][k] + matrix[seq[j]][k]
            else:
                C[j][k] = max(C[j - 1][k], C[j][k - 1]) + matrix[seq[j]][k]
    return C[-1][-1]