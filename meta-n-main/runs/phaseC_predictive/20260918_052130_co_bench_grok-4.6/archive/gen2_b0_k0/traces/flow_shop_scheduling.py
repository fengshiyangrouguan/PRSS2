def solve(**kwargs):
    from itertools import permutations
    n = kwargs['n']
    m = kwargs['m']
    matrix = kwargs['matrix']
    if n == 0:
        return {'job_sequence': []}
    jobs = list(range(n))
    jobs.sort(key=lambda j: sum(matrix[j]), reverse=True)
    if n <= 3:
        first = jobs
    else:
        first = jobs[:3]
    best_seed = min(permutations(first), key=lambda p: makespan(p, matrix))
    seq = list(best_seed)
    remaining = jobs[len(first):]
    for job in remaining:
        n_current = len(seq)
        best_pos = 0
        best_ms = float('inf')
        for pos in range(n_current + 1):
            new_seq = seq[:pos] + [job] + seq[pos:]
            ms = makespan(new_seq, matrix)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq = seq[:best_pos] + [job] + seq[best_pos:]
    return {'job_sequence': [x + 1 for x in seq]}

def makespan(seq, matrix):
    n = len(seq)
    m = len(matrix[0])
    C = [[0] * m for _ in range(n)]
    for j in range(m):
        C[0][j] = C[0][j - 1] + matrix[seq[0]][j] if j > 0 else matrix[seq[0]][0]
    for i in range(1, n):
        C[i][0] = C[i - 1][0] + matrix[seq[i]][0]
        for j in range(1, m):
            C[i][j] = max(C[i - 1][j], C[i][j - 1]) + matrix[seq[i]][j]
    return C[n - 1][m - 1]