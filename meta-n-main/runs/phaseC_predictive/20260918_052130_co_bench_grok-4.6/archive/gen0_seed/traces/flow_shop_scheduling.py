def solve(**kwargs):
    n = kwargs['n']
    m = kwargs['m']
    matrix = kwargs['matrix']
    if n == 0:
        return {'job_sequence': []}
    if n == 1:
        return {'job_sequence': [1]}
    total = [sum(row) for row in matrix]
    jobs = sorted(range(n), key=lambda i: total[i], reverse=True)
    def makespan(seq):
        C = [[0] * m for _ in range(len(seq))]
        for j in range(m):
            C[0][j] = C[0][j - 1] + matrix[seq[0]][j] if j > 0 else matrix[seq[0]][0]
        for i in range(1, len(seq)):
            C[i][0] = C[i - 1][0] + matrix[seq[i]][0]
            for j in range(1, m):
                C[i][j] = max(C[i - 1][j], C[i][j - 1]) + matrix[seq[i]][j]
        return C[-1][-1]
    current = jobs[:2]
    best_makespan = makespan(current)
    best_seq = current[:]
    for k in range(2, n):
        job = jobs[k]
        best_insert_makespan = float('inf')
        best_insert_pos = 0
        for pos in range(k + 1):
            temp = current[:pos] + [job] + current[pos:]
            cm = makespan(temp)
            if cm < best_insert_makespan:
                best_insert_makespan = cm
                best_insert_pos = pos
        current = current[:best_insert_pos] + [job] + current[best_insert_pos:]
        if best_insert_makespan < best_makespan:
            best_makespan = best_insert_makespan
            best_seq = current[:]
    job_sequence = [i + 1 for i in best_seq]
    return {'job_sequence': job_sequence}