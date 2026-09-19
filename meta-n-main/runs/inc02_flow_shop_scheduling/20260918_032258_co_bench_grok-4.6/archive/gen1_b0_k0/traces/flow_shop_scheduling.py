import itertools

def solve(**kwargs):
    n = kwargs['n']
    m = kwargs['m']
    matrix = kwargs['matrix']
    if n <= 9:
        seq = list(range(n))
        best_ms = float('inf')
        best_perm = None
        for p in itertools.permutations(seq):
            C = [[0] * m for _ in range(n)]
            for i in range(n):
                job = p[i]
                for k in range(m):
                    if i == 0:
                        C[i][k] = C[i][k - 1] if k > 0 else 0 + matrix[job][k]
                    else:
                        if k == 0:
                            C[i][k] = C[i - 1][k] + matrix[job][k]
                        else:
                            C[i][k] = max(C[i - 1][k], C[i][k - 1]) + matrix[job][k]
            ms = C[n - 1][m - 1]
            if ms < best_ms:
                best_ms = ms
                best_perm = p
        return {'job_sequence': [x + 1 for x in best_perm]}
    else:
        return {'job_sequence': list(range(1, n + 1))}