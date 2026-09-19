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
            for i in range(n):
                if i == 0 and j == 0:
                    C[i][j] = matrix[seq[i]][j]
                elif i == 0:
                    C[i][j] = C[i][j-1] + matrix[seq[i]][j]
                elif j == 0:
                    C[i][j] = C[i-1][j] + matrix[seq[i]][j]
                else:
                    C[i][j] = matrix[seq[i]][j] + max(C[i-1][j], C[i][j-1])
        return C[n-1][m-1]
    if n == 1:
        seq = [jobs[0]]
    else:
        s1 = [jobs[0], jobs[1]]
        s2 = [jobs[1], jobs[0]]
        ms1 = makespan(s1)
        ms2 = makespan(s2)
        seq = s1 if ms1 < ms2 else s2
        for k in range(2, n):
            job = jobs[k]
            best_ms = float('inf')
            best_seq = None
            for pos in range(len(seq) + 1):
                new_seq = seq[:pos] + [job] + seq[pos:]
                ms = makespan(new_seq)
                if ms < best_ms:
                    best_ms = ms
                    best_seq = new_seq
            seq = best_seq
    job_sequence = [j + 1 for j in seq]
    return {'job_sequence': job_sequence}