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
        for i in range(n):
            j = seq[i]
            p = matrix[j]
            if i == 0:
                C[i][0] = p[0]
                for jj in range(1, m):
                    C[i][jj] = C[i][jj - 1] + p[jj]
            else:
                C[i][0] = C[i - 1][0] + p[0]
                for jj in range(1, m):
                    C[i][jj] = max(C[i - 1][jj], C[i][jj - 1]) + p[jj]
        return C[n - 1][m - 1]
    if n == 1:
        return {'job_sequence': [1]}
    seq = jobs[:2]
    if makespan(seq) > makespan(seq[::-1]):
        seq = seq[::-1]
    for k in range(2, n):
        best_m = float('inf')
        best_pos = 0
        best_seq = None
        for pos in range(k + 1):
            temp = seq[:pos] + [jobs[k]] + seq[pos:]
            cm = makespan(temp)
            if cm < best_m:
                best_m = cm
                best_pos = pos
                best_seq = temp
        seq = best_seq
    perm = [jobs.index(j) + 1 for j in seq]
    return {'job_sequence': perm}