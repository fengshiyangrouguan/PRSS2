import itertools
import time

def solve(**kwargs):
    n = kwargs['n']
    m = kwargs['m']
    matrix = kwargs['matrix']
    if n <= 5:
        best_makespan = float('inf')
        best_sequence = None
        for perm in itertools.permutations(range(n)):
            makespan = compute_makespan(perm, matrix)
            if makespan < best_makespan:
                best_makespan = makespan
                best_sequence = perm
        return {'job_sequence': [x + 1 for x in best_sequence]}
    else:
        return neh_heuristic(matrix)