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
    # Compute makespan for a given job sequence (0-indexed)
    def compute_makespan(seq):
        C = [[0] * m for _ in range(n)]
        for j in range(n):
            for k in range(m):
                if j == 0 and k == 0:
                    C[j][k] = matrix[seq[j]][k]
                elif j == 0:
                    C[j][k] = C[j][k-1] + matrix[seq[j]][k]
                elif k == 0:
                    C[j][k] = C[j-1][k] + matrix[seq[j]][k]
                else:
                    C[j][k] = C[j-1][k] + C[j][k-1] - C[j-1][k-1] + matrix[seq[j]][k]
        return C[n-1][m-1]
    # NEH: start with first two jobs
    seq = [jobs[0], jobs[1]]
    ms1 = compute_makespan(seq)
    ms2 = compute_makespan([jobs[1], jobs[0]])
    if ms2 < ms1:
        seq = [jobs[1], jobs[0]]
    # Insert remaining jobs
    for i in range(2, n):
        job = jobs[i]
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(seq) + 1):
            temp_seq = seq[:pos] + [job] + seq[pos:]
            ms = compute_makespan(temp_seq)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq = seq[:best_pos] + [job] + seq[best_pos:]
    # Return 1-indexed permutation
    return {'job_sequence': [j + 1 for j in seq]}