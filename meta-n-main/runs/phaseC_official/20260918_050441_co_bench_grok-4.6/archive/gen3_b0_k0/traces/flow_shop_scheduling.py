def solve(**kwargs):
    n = kwargs['n']
    m = kwargs['m']
    matrix = kwargs['matrix']
    if n == 0:
        return {'job_sequence': []}
    if n == 1:
        return {'job_sequence': [1]}
    totals = [sum(row) for row in matrix]
    jobs = list(range(n))
    jobs.sort(key=lambda x: totals[x], reverse=True)
    def makespan(seq):
        seq = [s - 1 for s in seq]
        C = [[0] * m for _ in range(n)]
        for j in range(m):
            C[0][j] = matrix[seq[0]][j] if j == 0 else C[0][j - 1] + matrix[seq[0]][j]
        for i in range(1, n):
            C[i][0] = C[i - 1][0] + matrix[seq[i]][0]
            for j in range(1, m):
                C[i][j] = min(C[i - 1][j], C[i][j - 1]) + matrix[seq[i]][j]
        return C[n - 1][m - 1]
    seq = [jobs[0] + 1, jobs[1] + 1]
    for k in range(2, n):
        best_makespan = float('inf')
        best_pos = 0
        job = jobs[k] + 1
        for pos in range(len(seq) + 1):
            temp_seq = seq[:pos] + [job] + seq[pos:]
            ms = makespan(temp_seq)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        seq = seq[:best_pos] + [job] + seq[best_pos:]
    return {'job_sequence': seq}