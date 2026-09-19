def compute_makespan(perm: list, matrix: list, n: int, m: int) -> int:
    """Compute makespan for a permutation of jobs in a flow-shop scheduling problem.

    Args:
        perm: list[int] — job indices in 0-based order (length n).
        matrix: list[list[int]] — processing times; shape (m, n) where matrix[k][j] is time of job j on machine k.
        n: int — number of jobs.
        m: int — number of machines.

    Returns:
        int — makespan (completion time of last job on last machine).
    """
    C = [[0] * m for _ in range(n + 1)]
    for j in range(1, n + 1):
        for k in range(m):
            C[j][k] = max(C[j - 1][k], C[j][k - 1]) + matrix[k][perm[j - 1]]
    return C[n][m - 1]