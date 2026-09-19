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
    # Function to compute makespan for a given sequence using classical recurrence
    def makespan(seq):
        C = [[0] * m for _ in range(n)]
        for j in range(m):
            C[0][j] = C[0][j - 1] + matrix[seq[0]][j] if j > 0 else matrix[seq[0]][0]
        for i in range(1, n):
            C[i][0] = C[i - 1][0] + matrix[seq[i]][0]
            for j in range(1, m):
                C[i][j] = max(C[i - 1][j], C[i][j - 1]) + matrix[seq[i]][j]
        return C[n - 1][m - 1]
    # Handle small cases
    if n == 1:
        return {'job_sequence': [1]}
    # Start with first two jobs in the sorted list and find best order
    best_seq = [jobs[0], jobs[1]]
    best_ms = makespan(best_seq)
    # Insert remaining jobs one by one in best position
    for k in range(2, n):
        job = jobs[k]
        best_insert = None
        best_ms_insert = float('inf')
        for pos in range(k + 1):
            temp_seq = best_seq[:pos] + [job] + best_seq[pos:]
            ms = makespan(temp_seq)
            if ms < best_ms_insert:
                best_ms_insert = ms
                best_insert = pos
        best_seq = best_seq[:best_insert] + [job] + best_seq[best_insert:]
    # Return 1-indexed job sequence
    return {'job_sequence': [j + 1 for j in best_seq]}