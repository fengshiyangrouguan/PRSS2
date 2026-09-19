def solve(**kwargs):
    n = kwargs['n']
    m = kwargs['m']
    matrix = kwargs['matrix']
    if n == 0:
        return {'job_sequence': []}
    # Compute total processing time for each job
    total = [sum(row) for row in matrix]
    # Sort jobs by total time descending
    jobs = sorted(range(n), key=lambda j: total[j], reverse=True)
    # Compute makespan for a given sequence
    def compute_makespan(seq):
        C = [[0] * m for _ in range(n)]
        for i in range(n):
            job = seq[i]
            for j in range(m):
                if i == 0 and j == 0:
                    C[i][j] = matrix[job][j]
                elif i == 0:
                    C[i][j] = C[i][j-1] + matrix[job][j]
                elif j == 0:
                    C[i][j] = C[i-1][j] + matrix[job][j]
                else:
                    C[i][j] = max(C[i-1][j], C[i][j-1]) + matrix[job][j]
        return C[n-1][m-1]
    # NEH heuristic
    seq = [jobs[0]]
    for j in jobs[1:]:
        best_pos = 0
        best_makespan = float('inf')
        for pos in range(len(seq) + 1):
            temp_seq = seq[:pos] + [j] + seq[pos:]
            makespan = compute_makespan(temp_seq)
            if makespan < best_makespan:
                best_makespan = makespan
                best_pos = pos
        seq.insert(best_pos, j)
    job_sequence = [x + 1 for x in seq]
    return {'job_sequence': job_sequence}
