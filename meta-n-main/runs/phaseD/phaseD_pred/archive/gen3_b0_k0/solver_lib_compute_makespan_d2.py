def compute_makespan(matrix: list[list[int]], perm: list[int]) -> int:
    """Compute makespan for a given job permutation on the flow-shop instance.

    Args:
        matrix: 2-D list; matrix[i][k] = processing time of job i on machine k (0-based).
        perm: list of job indices in the order processed.

    Returns:
        int: makespan (total completion time of the last job on the last machine).
    """
    n = len(matrix)
    m = len(matrix[0])
    C = [[0] * m for _ in range(n + 1)]
    for j in range(1, n + 1):
        for k in range(m):
            if k == 0:
                C[j][k] = C[j - 1][k] + matrix[perm[j - 1]][k]
            else:
                C[j][k] = max(C[j - 1][k], C[j][k - 1]) + matrix[perm[j - 1]][k]
    return C[n][m - 1]