def solve(**kwargs):
    n = kwargs['n']
    m = kwargs['m']
    matrix = kwargs['matrix']
    if n == 0:
        return {'job_sequence': []}
    # Compute total processing time for each job
    total_times = [sum(row) for row in matrix]
    # Sort jobs by total time descending
    jobs = list(range(n))
    jobs.sort(key=lambda j: total_times[j], reverse=True)
    # NEH heuristic
    def makespan(perm):
        C = [[0] * m for _ in range(n)]
        for j in range(n):
            job = perm[j]
            for k in range(m):
                if j == 0 and k == 0:
                    C[j][k] = matrix[job][k]
                elif j == 0:
                    C[j][k] = C[j][k-1] + matrix[job][k]
                elif k == 0:
                    C[j][k] = C[j-1][k] + matrix[job][k]
                else:
                    C[j][k] = max(C[j-1][k], C[j][k-1]) + matrix[job][k]
        return C[n-1][m-1]
    # Start with first two jobs
    if n == 1:
        best_perm = [0]
    else:
        best_perm = [jobs[0], jobs[1]]
        best_ms = makespan(best_perm)
        if makespan([jobs[1], jobs[0]]) < best_ms:
            best_perm = [jobs[1], jobs[0]]
    # Insert remaining jobs
    for i in range(2, n):
        job = jobs[i]
        best_insert = None
        best_ms = float('inf')
        for pos in range(len(best_perm) + 1):
            temp_perm = best_perm[:pos] + [job] + best_perm[pos:]
            ms = makespan(temp_perm)
            if ms < best_ms:
                best_ms = ms
                best_insert = pos
        best_perm = best_perm[:best_insert] + [job] + best_perm[best_insert:]
    # Convert to 1-indexed
    job_sequence = [j + 1 for j in best_perm]
    return {'job_sequence': job_sequence}