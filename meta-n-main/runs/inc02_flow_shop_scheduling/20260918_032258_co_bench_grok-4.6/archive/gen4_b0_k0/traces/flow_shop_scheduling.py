def solve(**kwargs):
    n = kwargs['n']
    m = kwargs['m']
    matrix = kwargs['matrix']
    def makespan(seq, matrix):
        C = [[0] * m for _ in range(n + 1)]
        for j in range(1, n + 1):
            job = seq[j - 1]
            p = matrix[job - 1]
            C[j][0] = C[j - 1][0] + p[0]
            for i in range(1, m):
                C[j][i] = max(C[j - 1][i], C[j][i - 1]) + p[i]
        return C[n][m]
    best_seq = None
    best_makespan = float('inf')
    for perm in itertools.permutations(range(n)):
        seq = [p + 1 for p in perm]
        ms = makespan(seq, matrix)
        if ms < best_makespan:
            best_makespan = ms
            best_seq = seq
    return {'job_sequence': best_seq}