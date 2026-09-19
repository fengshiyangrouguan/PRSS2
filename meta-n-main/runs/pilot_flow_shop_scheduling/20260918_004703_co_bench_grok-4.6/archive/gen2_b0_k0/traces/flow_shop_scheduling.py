def solve(**kwargs):
    n = kwargs['n']
    m = kwargs['m']
    matrix = kwargs['matrix']
    if n == 0:
        return {'job_sequence': []}
    # Define compute_makespan helper
    def compute_makespan(seq, matrix, m):
        C = [[0] * m for _ in range(n)]
        for j in range(m):
            C[0][j] = sum(matrix[i][j] for i in seq[:1])
        for i in range(1, n):
            C[i][0] = C[i-1][0] + matrix[seq[i]][0]
            for j in range(1, m):
                C[i][j] = max(C[i-1][j], C[i][j-1]) + matrix[seq[i]][j]
        return C[n-1][m-1]
    # NEH heuristic
    total_times = [sum(row) for row in matrix]
    jobs = sorted(range(n), key=lambda i: total_times[i], reverse=True)
    if n == 1:
        return {'job_sequence': [1]}
    seq = jobs[:2]
    for job in jobs[2:]:
        best_makespan = float('inf')
        best_pos = 0
        for pos in range(len(seq) + 1):
            new_seq = seq[:pos] + [job] + seq[pos:]
            makespan = compute_makespan(new_seq, matrix, m)
            if makespan < best_makespan:
                best_makespan = makespan
                best_pos = pos
        seq = seq[:best_pos] + [job] + seq[best_pos:]
    job_sequence = [j + 1 for j in seq]
    return {'job_sequence': job_sequence}