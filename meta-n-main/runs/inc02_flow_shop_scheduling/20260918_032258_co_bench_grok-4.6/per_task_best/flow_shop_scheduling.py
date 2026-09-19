import itertools
import time

def solve(**kwargs):
    n = kwargs['n']
    m = kwargs['m']
    matrix = kwargs['matrix']
    if n > 10:
        return {'job_sequence': list(range(1, n + 1))}
    start_time = time.time()
    best_makespan = float('inf')
    best_seq = None
    for perm in itertools.permutations(range(n)):
        if time.time() - start_time > 8.0:
            break
        C = [[0] * m for _ in range(n)]
        for i in range(n):
            for j in range(m):
                if i == 0 and j == 0:
                    C[i][j] = matrix[perm[i]][j]
                elif i == 0:
                    C[i][j] = C[i][j - 1] + matrix[perm[i]][j]
                elif j == 0:
                    C[i][j] = C[i - 1][j] + matrix[perm[i]][j]
                else:
                    C[i][j] = max(C[i - 1][j], C[i][j - 1]) + matrix[perm[i]][j]
        ms = C[n - 1][m - 1]
        if ms < best_makespan:
            best_makespan = ms
            best_seq = perm
    if best_seq is None:
        best_seq = tuple(range(n))
    return {'job_sequence': [x + 1 for x in best_seq]}