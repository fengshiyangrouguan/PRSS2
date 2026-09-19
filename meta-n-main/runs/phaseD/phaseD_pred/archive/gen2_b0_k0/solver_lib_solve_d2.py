def solve(**kwargs):
    """Exact permutation solver for flow shop scheduling (Fm||Cmax) with fixed DP.

    Args:
        n: int — number of jobs.
        m: int — number of machines.
        matrix: list[list[float]] — processing time matrix (job x machine).

    Returns:
        dict: {'job_sequence': list of 1-based job indices in optimal order}
    """
    import itertools
    from math import inf

    matrix = kwargs['matrix']
    best_makespan = inf
    best_sequence = None

    for perm in itertools.permutations(range(kwargs['n'])):
        makespan = compute_makespan(perm, matrix)
        if makespan < best_makespan:
            best_makespan = makespan
            best_sequence = perm

    return {'job_sequence': [x + 1 for x in best_sequence]}