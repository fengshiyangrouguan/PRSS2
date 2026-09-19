def solve(**kwargs):
    n = kwargs['n']
    m = kwargs['m']
    matrix = kwargs['matrix']
    if n == 0:
        return {'job_sequence': []}
    totals = [sum(row) for row in matrix]
    jobs = list(range(n))
    jobs.sort(key=lambda x: totals[x], reverse=True)
    if n == 1:
        return {'job_sequence': [1]}
    def makespan(seq):
        p = [[matrix[j][k] for k in range(m)] for j in seq]
        C = [[0] * m for _ in range(n)]
        for j in range(m):
            C[0][j] = C[0][j-1] + p[0][j] if j > 0 else p[0][0]
        for i in range(1, n):
            C[i][0] = C[i-1][0] + p[i][0]
        for j in range(1, m):
            for i in range(1, n):
                C[i][j] = max(C[i-1][j], C[i][j-1]) + p[i][j]
        return C[n-1][m-1]
    best_seq = jobs[:2]
    ms1 = makespan(best_seq)
    best_seq = jobs[1:3]
    ms2 = makespan(best_seq)
    if ms1 < ms2:
        best_seq = jobs[:2]
    else:
        best_seq = jobs[1:3]
    for k in range(2, n):
        job = jobs[k]
        best_insert = None
        best_ms = float('inf')
        for pos in range(len(best_seq) + 1):
            seq = best_seq[:pos] + [job] + best_seq[pos:]
            ms = makespan(seq)
            if ms < best_ms:
                best_ms = ms
                best_insert = pos
        best_seq = best_seq[:best_insert] + [job] + best_seq[best_insert:]
    return {'job_sequence': [x + 1 for x in best_seq]}