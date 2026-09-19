def solve(**kwargs):
    n = kwargs['n']
    m = kwargs['m']
    matrix = kwargs['matrix']
    job_sequence = solver_lib.best_approximation_heuristic(n, m, matrix)
    return {'job_sequence': [j + 1 for j in job_sequence]}