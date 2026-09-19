import solver_lib

def solve(**kwargs):
    matrix = kwargs['matrix']
    neh_0based = solver_lib.neh_heuristic(matrix)
    job_sequence = [j + 1 for j in neh_0based]
    return {'job_sequence': job_sequence}