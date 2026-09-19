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
    # Compute makespan for a given sequence (0-based)
    def compute_makespan(seq):
        times = [0] * m
        for job in seq:
            for k in range(m):
                times[k] += matrix[job][k]
                if k > 0:
                    times[k] = max(times[k], times[k-1])
        return times[-1]
    # Start with the first job
    current_seq = [jobs[0]]
    # Insert second job
    if n > 1:
        best_makespan = float('inf')
        best_pos = 0
        for pos in range(2):
            temp_seq = current_seq[:]
            temp_seq.insert(pos, jobs[1])
            makespan = compute_makespan(temp_seq)
            if makespan < best_makespan:
                best_makespan = makespan
                best_pos = pos
        current_seq.insert(best_pos, jobs[1])
    # Insert remaining jobs
    for i in range(2, n):
        best_makespan = float('inf')
        best_pos = 0
        job = jobs[i]
        curr_len = len(current_seq)
        for pos in range(curr_len + 1):
            temp_seq = current_seq[:]
            temp_seq.insert(pos, job)
            makespan = compute_makespan(temp_seq)
            if makespan < best_makespan:
                best_makespan = makespan
                best_pos = pos
        current_seq.insert(best_pos, job)
    # Convert to 1-indexed
    job_sequence = [j + 1 for j in current_seq]
    return {'job_sequence': job_sequence}