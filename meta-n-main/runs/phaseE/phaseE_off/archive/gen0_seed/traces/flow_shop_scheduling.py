import itertools
from math import inf

def solve(**kwargs):
    n = kwargs['n']
    m = kwargs['m']
    matrix = kwargs['matrix']
    best_makespan = inf
    best_sequence = None
    for perm in itertools.permutations(range(n)):
        C = [[0] * m for _ in range(n + 1)]
        for j in range(1, n + 1):
            for k in range(m):
                C[j][k] = max(C[j-1][k], C[j][k-1]) + matrix[perm[j-1]][k]
        makespan = C[n][m]
        if makespan < best_makespan:
            best_makespan = makespan
            best_sequence = perm
    return {'job_sequence': [x + 1 for x in best_sequence]}