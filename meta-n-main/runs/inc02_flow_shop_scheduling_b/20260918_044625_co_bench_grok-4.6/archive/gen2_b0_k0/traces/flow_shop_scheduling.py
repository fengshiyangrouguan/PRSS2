def solve(**kwargs):
    n = kwargs['n']
    m = kwargs['m']
    matrix = kwargs['matrix']
    if n == 0:
        return {'job_sequence': []}
    # Compute sum of processing times for each job
    sums = [sum(row) for row in matrix]
    # Sort jobs by sum descending
    sorted_jobs = sorted(range(n), key=lambda x: sums[x], reverse=True)
    if n == 1:
        return {'job_sequence': [1]}
    # Start with first two jobs
    partial = [sorted_jobs[0], sorted_jobs[1]]
    # For each subsequent job
    for job in sorted_jobs[2:]:
        best_makespan = float('inf')
        best_pos = 0
        # Try all insertion positions
        for pos in range(len(partial) + 1):
            temp_seq = partial[:pos] + [job] + partial[pos:]
            # Compute makespan
            C = [[0] * m for _ in range(n)]
            for j in range(m):
                for i in range(n):
                    if j == 0:
                        C[i][j] = matrix[temp_seq[i]][j]
                    else:
                        C[i][j] = matrix[temp_seq[i]][j] + max(C[i - 1][j] if i > 0 else 0, C[i][j - 1] if j > 0 else 0)
            ms = C[n - 1][m - 1]
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        # Insert job at best position
        partial = partial[:best_pos] + [job] + partial[best_pos:]
    # Convert to 1-based indices
    job_sequence = [x + 1 for x in partial]
    return {'job_sequence': job_sequence}