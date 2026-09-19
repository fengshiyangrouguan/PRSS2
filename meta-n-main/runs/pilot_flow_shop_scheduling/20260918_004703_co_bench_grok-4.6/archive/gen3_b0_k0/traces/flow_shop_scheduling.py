def solve(**kwargs):
    n = kwargs['n']
    m = kwargs['m']
    matrix = kwargs['matrix']
    if n == 0:
        return {'job_sequence': []}
    # Compute sum of processing times for each job
    sums = [sum(row) for row in matrix]
    # Sort jobs by decreasing sum
    jobs = list(range(n))
    jobs.sort(key=lambda j: sums[j], reverse=True)
    # Start with the first job
    seq = [jobs[0]]
    for i in range(1, n):
        best_makespan = float('inf')
        best_pos = 0
        for pos in range(len(seq) + 1):
            temp_seq = seq[:pos] + [jobs[i]] + seq[pos:]
            makespan = solver_lib.compute_makespan(temp_seq, matrix, n, m)
            if makespan < best_makespan:
                best_makespan = makespan
                best_pos = pos
        seq.insert(best_pos, jobs[i])
    return {'job_sequence': [j + 1 for j in seq]}