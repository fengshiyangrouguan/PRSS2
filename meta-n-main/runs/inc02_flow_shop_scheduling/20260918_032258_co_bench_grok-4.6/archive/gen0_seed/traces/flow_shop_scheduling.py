def solve(**kwargs):
    n = kwargs['n']
    m = kwargs['m']
    matrix = kwargs['matrix']
    if n == 0:
        return {'job_sequence': []}
    jobs = list(range(n))
    total = [sum(row) for row in matrix]
    jobs.sort(key=lambda j: total[j], reverse=True)
    if n == 1:
        return {'job_sequence': [1]}
    def makespan(seq):
        C = [[0] * m for _ in range(n)]
        for j in range(m):
            C[0][j] = C[0][j-1] + matrix[seq[0]][j] if j > 0 else matrix[seq[0]][0]
        for i in range(1, n):
            C[i][0] = C[i-1][0] + matrix[seq[i]][0]
            for j in range(1, m):
                C[i][j] = max(C[i-1][j], C[i][j-1]) + matrix[seq[i]][j]
        return C[n-1][m-1]
    current = jobs[:2]
    if n == 2:
        if makespan([jobs[0], jobs[1]]) < makespan([jobs[1], jobs[0]]):
            current = [jobs[0], jobs[1]]
        else:
            current = [jobs[1], jobs[0]]
    else:
        for k in range(2, n):
            best_m = float('inf')
            best_pos = 0
            for pos in range(k + 1):
                temp = current[:pos] + [jobs[k]] + current[pos:]
                mval = makespan(temp)
                if mval < best_m:
                    best_m = mval
                    best_pos = pos
            current = current[:best_pos] + [jobs[k]] + current[best_pos:]
    return {'job_sequence': [j + 1 for j in current]}