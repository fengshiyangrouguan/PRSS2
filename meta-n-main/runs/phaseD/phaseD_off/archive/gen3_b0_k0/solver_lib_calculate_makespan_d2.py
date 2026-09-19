def calculate_makespan(matrix: list[list[int]], perm: tuple[int, ...]) -> int:
    """Compute makespan for a flow-shop schedule.

    Args:
        matrix: 2D list where matrix[j][k] is processing time of job j on machine k (0-based)
        perm: tuple of job indices in the sequence order (0-based)

    Returns:
        int: makespan (completion time of the last job on the last machine)
    """
    n = len(perm)
    m = len(matrix[0]) if matrix else 0
    C = [[0] * m for _ in range(n + 1)]
    for j in range(1, n + 1):
        for k in range(m):
            if k == 0:
                C[j][k] = C[j - 1][k] + matrix[perm[j - 1]][k]
            elif j == 1:
                C[j][k] = C[j][k - 1] + matrix[perm[j - 1]][k]
            else:
                C[j][k] = max(C[j - 1][k], C[j][k - 1]) + matrix[perm[j - 1]][k]
    return C[n][m - 1]