def solve(**kwargs):
    n = kwargs['n']
    m = kwargs['m']
    matrix = kwargs['matrix']
    if n == 0:
        return {'job_sequence': []}
    total_times = [sum(row) for row in matrix]
    jobs = list(range(n))
    jobs.sort(key=lambda j: total_times[j], reverse=True)
    if n == 1:
        return {'job_sequence': [1]}
    seq = jobs[:2]
    for j in jobs[2:]:
        best_makespan = float('inf')
        best_pos = 0
        for pos in range(len(seq) + 1):
            temp_seq = seq[:pos] + [j] + seq[pos:]
            makespan = solver_lib.compute_makespan(temp_seq, matrix, n, m)
            if makespan < best_makespan:
                best_makespan = makespan
                best_pos = pos
        seq.insert(best_pos, j)
    return {'job_sequence': [x + 1 for x in seq]}