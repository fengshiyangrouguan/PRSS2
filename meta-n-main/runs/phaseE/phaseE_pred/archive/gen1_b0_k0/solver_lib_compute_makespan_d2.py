def compute_makespan(n: int, m: int, matrix: list[list[int]], perm: list[int]) -> int:
    """Compute makespan for a single permutation in the flow-shop scheduling problem.

    Args:
        n: number of jobs
        m: number of machines
        matrix: n x m matrix where matrix[j][k] = processing time of job (j+1) on machine (k+1)
        perm: 0-based permutation of job indices

    Returns:
        makespan: scalar makespan value
    """
    C = [[0] * m for _ in range(n + 1)]
    for j in range(1, n + 1):
        for k in range(m):
            if k == 0:
                C[j][k] = C[j - 1][k] + matrix[perm[j - 1]][k]
            else:
                C[j][k] = max(C[j - 1][k], C[j][k - 1]) + matrix[perm[j - 1]][k]
    return C[n][m]

def solve(**kwargs):
    """Wrapper that re-uses the exact structure of the original script but replaces the buggy DP."""
    n = kwargs['n']
    m = kwargs['m']
    matrix = kwargs['matrix']
    best_makespan = float('inf')
    best_sequence = None
    for perm in itertools.permutations(range(n)):
        makespan = compute_makespan(n, m, matrix, perm)
        if makespan < best_makespan:
            best_makespan = makespan
            best_sequence = perm
    return {'job_sequence': [x + 1 for x in best_sequence]}