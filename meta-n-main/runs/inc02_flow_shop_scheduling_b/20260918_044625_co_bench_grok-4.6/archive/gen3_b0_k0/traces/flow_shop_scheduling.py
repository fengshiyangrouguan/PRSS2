def solve(**kwargs):
    n = kwargs['n']
    m = kwargs['m']
    matrix = kwargs['matrix']
    if n == 0:
        return {'job_sequence': []}
    # Compute total processing time for each job
    totals = [sum(row) for row in matrix]
    # Sort jobs by total time descending
    jobs = list(range(n))
    jobs.sort(key=lambda j: totals[j], reverse=True)
    # Helper to compute makespan for a given sequence
    def compute_makespan(seq):
        C = [[0] * m for _ in range(n)]
        for i in range(n):
            for j in range(m):
                if i == 0 and j == 0:
                    C[i][j] = matrix[seq[i]][j]
                elif i == 0:
                    C[i][j] = C[i][j-1] + matrix[seq[i]][j]
                elif j == 0:
                    C[i][j] = C[i-1][j] + matrix[seq[i]][j]
                else:
                    C[i][j] = max(C[i-1][j], C[i][j-1]) + matrix[seq[i]][j]
        return C[n-1][m-1]
    # Start with first two jobs
    if n == 1:
        return {'job_sequence': [1]}
    seq = jobs[:2]
    # For each subsequent job, find best insertion position
    for k in range(2, n):
        job = jobs[k]
        best_makespan = float('inf')
        best_pos = 0
        for pos in range(len(seq) + 1):
            temp_seq = seq[:pos] + [job] + seq[pos:]
            makespan = compute_makespan(temp_seq)
            if makespan < best_makespan:
                best_makespan = makespan
                best_pos = pos
        seq = seq[:best_pos] + [job] + seq[best_pos:]
    # Convert to 1-based indices
    job_sequence = [j + 1 for j in seq]
    return {'job_sequence': job_sequence}