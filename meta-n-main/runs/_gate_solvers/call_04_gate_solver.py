def solve(**kwargs):
    n = kwargs['n']
    m = kwargs['m']
    matrix = kwargs['matrix']
    if n == 0:
        return {'job_sequence': []}
    totals = [sum(row) for row in matrix]
    jobs = list(range(n))
    jobs.sort(key=lambda j: totals[j], reverse=True)
    def makespan(seq):
        C = [[0] * m for _ in range(n)]
        for j in range(m):
            C[0][j] = (C[0][j-1] + matrix[seq[0]][j]) if j > 0 else matrix[seq[0]][0]
        for i in range(1, n):
            C[i][0] = C[i-1][0] + matrix[seq[i]][0]
            for j in range(1, m):
                C[i][j] = max(C[i-1][j], C[i][j-1]) + matrix[seq[i]][j]
        return C[n-1][m-1]
    if n == 1:
        return {'job_sequence': [1]}
    seq = [jobs[0], jobs[1]]
    if makespan([jobs[1], jobs[0]]) < makespan(seq):
        seq = [jobs[1], jobs[0]]
    for k in range(2, n):
        job = jobs[k]
        best_m = makespan(seq)
        best_pos = 0
        for pos in range(len(seq) + 1):
            new_seq = seq[:pos] + [job] + seq[pos:]
            m_new = makespan(new_seq)
            if m_new < best_m:
                best_m = m_new
                best_pos = pos
        seq = seq[:best_pos] + [job] + seq[best_pos:]
    job_sequence = [jobs.index(j) + 1 for j in seq]
    return {'job_sequence': job_sequence}
