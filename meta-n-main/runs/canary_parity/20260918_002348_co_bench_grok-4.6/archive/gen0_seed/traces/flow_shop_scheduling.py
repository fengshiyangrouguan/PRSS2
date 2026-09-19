import itertools

def solve(**kwargs):
    n = kwargs.get('n', 0)
    m = kwargs.get('m', 0)
    matrix = kwargs.get('matrix', [])
    if n == 0:
        return {'job_sequence': []}
    if n > 8:
        return {'job_sequence': list(range(1, n + 1))}
    best = float('inf')
    best_perm = None
    for p in itertools.permutations(range(n)):
        c = [[0] * m for _ in range(n)]
        for i, job in enumerate(p):
            for j in range(m):
                c[job][j] = matrix[job][j] + (c[job][j - 1] if j > 0 else 0)
                if i > 0:
                    prev_job = p[i - 1]
                    c[job][j] = max(c[job][j], c[prev_job][j])
        ms = c[p[-1]][m - 1]
        if ms < best:
            best = ms
            best_perm = p
    return {'job_sequence': [x + 1 for x in best_perm]}