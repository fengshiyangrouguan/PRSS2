def solve(**kwargs):
    n = kwargs['n']
    m = kwargs['m']
    matrix = kwargs['matrix']
    if m == 2:
        perm = solver_lib.johnson_heuristic(matrix)
        job_sequence = [j + 1 for j in perm]
    else:
        jobs = list(range(n))
        jobs.sort(key=lambda j: sum(matrix[j]))
        job_sequence = [j + 1 for j in jobs]
    return {'job_sequence': job_sequence}