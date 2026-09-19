def solve(**kwargs):
    n = kwargs['n']
    m = kwargs['m']
    matrix = kwargs['matrix']
    if n == 0:
        return {'job_sequence': []}
    if n == 1:
        return {'job_sequence': [1]}
    # compute total processing times
    totals = [sum(row) for row in matrix]
    jobs = list(range(n))
    jobs.sort(key=lambda j: totals[j], reverse=True)
    # nested function to compute makespan using classical recurrence
    def compute_makespan(seq):
        C = [[0] * m for _ in range(n)]
        for j in range(m):
            C[0][j] = sum(matrix[seq[i]][j] for i in range(j + 1))
        for i in range(1, n):
            C[i][0] = sum(matrix[seq[k]][0] for k in range(i + 1))
        for i in range(1, n):
            for j in range(1, m):
                C[i][j] = max(C[i - 1][j], C[i][j - 1]) + matrix[seq[i]][j]
        return C[n - 1][m - 1]
    # NEH heuristic
    seq = [jobs[0]]
    for k in range(1, n):
        best_makespan = float('inf')
        best_pos = 0
        for pos in range(len(seq) + 1):
            temp_seq = seq[:pos] + [jobs[k]] + seq[pos:]
            makespan = compute_makespan(temp_seq)
            if makespan < best_makespan:
                best_makespan = makespan
                best_pos = pos
        seq.insert(best_pos, jobs[k])
    return {'job_sequence': [j + 1 for j in seq]}