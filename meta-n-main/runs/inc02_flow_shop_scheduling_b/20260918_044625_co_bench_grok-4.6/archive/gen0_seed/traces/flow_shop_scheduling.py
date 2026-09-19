import itertools

def solve(**kwargs):
    n = kwargs['n']
    m = kwargs['m']
    matrix = kwargs['matrix']
    if n == 0:
        return {'job_sequence': []}
    best_makespan = float('inf')
    best_seq = None
    for perm in itertools.permutations(range(1, n + 1)):
        seq = list(perm)
        C = [[0] * m for _ in range(n)]
        for j in range(m):
            for i in range(n):
                job = seq[i] - 1
                if j == 0:
                    C[i][j] = matrix[job][j]
                else:
                    prev = 0 if i == 0 else C[i - 1][j]
                    C[i][j] = max(prev, C[i][j - 1]) + matrix[job][j]
        makespan = C[n - 1][m - 1]
        if makespan < best_makespan:
            best_makespan = makespan
            best_seq = seq
    return {'job_sequence': best_seq}