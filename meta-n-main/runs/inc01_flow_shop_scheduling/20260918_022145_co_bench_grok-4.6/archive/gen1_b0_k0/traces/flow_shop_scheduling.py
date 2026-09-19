def solve(**kwargs):
    n = kwargs['n']
    m = kwargs['m']
    matrix = kwargs['matrix']
    # Compute total processing time for each job
    totals = [sum(row) for row in matrix]
    # Sort jobs by total time descending
    jobs = list(range(n))
    jobs.sort(key=lambda j: totals[j], reverse=True)
    # NEH heuristic
    def compute_makespan(seq):
        C = [[0] * m for _ in range(n)]
        for j in range(m):
            for i in range(n):
                p = matrix[seq[i]][j]
                if i == 0 and j == 0:
                    C[i][j] = p
                elif i == 0:
                    C[i][j] = C[i][j-1] + p
                elif j == 0:
                    C[i][j] = C[i-1][j] + p
                else:
                    C[i][j] = max(C[i-1][j], C[i][j-1]) + p
        return C[n-1][m-1]
    # Start with first two jobs
    seq = jobs[:2]
    for j in jobs[2:]:
        best_makespan = float('inf')
        best_pos = 0
        for pos in range(len(seq) + 1):
            temp_seq = seq[:pos] + [j] + seq[pos:]
            ms = compute_makespan(temp_seq)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        seq = seq[:best_pos] + [j] + seq[best_pos:]
    return {'job_sequence': [x + 1 for x in seq]}