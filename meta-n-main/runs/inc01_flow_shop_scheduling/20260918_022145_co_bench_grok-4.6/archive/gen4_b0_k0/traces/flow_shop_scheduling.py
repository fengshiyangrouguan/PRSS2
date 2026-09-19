import itertools

def solve(**kwargs):
    n = kwargs['n']
    m = kwargs['m']
    matrix = kwargs['matrix']
    if n > 10:
        return {'job_sequence': list(range(1, n + 1))}
    best_makespan = float('inf')
    best_seq = None
    for perm in itertools.permutations(range(n)):
        C = [[0] * m for _ in range(n)]
        for j in range(m):
            C[0][j] = C[0][j - 1] + matrix[perm[0]][j] if j > 0 else matrix[perm[0]][j]
        for i in range(1, n):
            for j in range(m):
                if j == 0:
                    C[i][j] = C[i - 1][j] + matrix[perm[i]][j]
                else:
                    C[i][j] = max(C[i - 1][j], C[i][j - 1]) + matrix[perm[i]][j]
        makespan = C[n - 1][m - 1]
        if makespan < best_makespan:
            best_makespan = makespan
            best_seq = perm
    return {'job_sequence': [j + 1 for j in best_seq]}