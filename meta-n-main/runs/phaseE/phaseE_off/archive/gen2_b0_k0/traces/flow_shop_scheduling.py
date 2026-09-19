def solve(**kwargs):
    n = kwargs['n']
    m = kwargs['m']
    matrix = kwargs['matrix']
    if n == 0:
        return {'job_sequence': []}
    # Compute total processing time for sorting
    total_time = [sum(row) for row in matrix]
    # Sort jobs by total time descending
    sorted_jobs = sorted(range(n), key=lambda i: total_time[i], reverse=True)
    # Function to compute makespan for a given permutation (0-based)
    def makespan(perm):
        C = [[0] * m for _ in range(n + 1)]
        for i in range(1, n + 1):
            job = perm[i - 1]
            for j in range(m):
                if j == 0:
                    C[i][j] = C[i - 1][j] + matrix[job][j]
                else:
                    C[i][j] = max(C[i - 1][j], C[i][j - 1]) + matrix[job][j]
        return C[n][m - 1]
    # NEH heuristic
    if n <= 2:
        best_perm = sorted_jobs
        best_ms = makespan(best_perm)
    else:
        # Start with first two jobs
        p1, p2 = sorted_jobs[0], sorted_jobs[1]
        ms1 = makespan([p1, p2])
        ms2 = makespan([p2, p1])
        if ms1 < ms2:
            current_perm = [p1, p2]
            best_ms = ms1
        else:
            current_perm = [p2, p1]
            best_ms = ms2
        # Insert remaining jobs
        for k in range(2, n):
            job = sorted_jobs[k]
            best_insert_ms = best_ms
            best_insert_pos = 0
            for pos in range(len(current_perm) + 1):
                new_perm = current_perm[:pos] + [job] + current_perm[pos:]
                new_ms = makespan(new_perm)
                if new_ms < best_insert_ms:
                    best_insert_ms = new_ms
                    best_insert_pos = pos
            current_perm = current_perm[:best_insert_pos] + [job] + current_perm[best_insert_pos:]
            best_ms = best_insert_ms
    # Convert to 1-based and return
    job_sequence = [j + 1 for j in current_perm]
    return {'job_sequence': job_sequence}