def solve(**kwargs):
    n = kwargs['n']
    m = kwargs['m']
    matrix = kwargs['matrix']
    if n <= 1:
        return {'job_sequence': [1]}
    from itertools import permutations
    best_makespan = float('inf')
    best_perm = None
    for p in permutations(range(n)):
        seq = [x + 1 for x in p]
        makespan = calculate_makespan(matrix, p)
        if makespan < best_makespan:
            best_makespan = makespan
            best_perm = p
    return {'job_sequence': list(best_perm) + [1]}