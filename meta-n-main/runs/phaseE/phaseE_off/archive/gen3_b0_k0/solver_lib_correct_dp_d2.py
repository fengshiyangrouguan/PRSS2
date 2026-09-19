def correct_dp(matrix: list, perm: list, n: int, m: int) -> float:
    """Compute the makespan for a given permutation of jobs on m machines.

    Args:
        matrix: 2D list of shape (n, m) containing processing times; job i is at index i.
        perm: list of n distinct job indices (0..n-1) in the order to process.
        n: number of jobs.
        m: number of machines.

    Returns:
        float: makespan (sum of completion times on the last machine) for the given perm, or None if n==0 or m==0.
    """
    if n == 0 or m == 0:
        return 0.0
    C = [[0.0] * m for _ in range(n + 1)]
    for j in range(1, n + 1):
        for k in range(m):
            if k == 0:
                C[j][k] = C[j - 1][k] + matrix[perm[j - 1]][k]
            else:
                C[j][k] = max(C[j - 1][k], C[j][k - 1]) + matrix[perm[j - 1]][k]
    return float(C[n][m])