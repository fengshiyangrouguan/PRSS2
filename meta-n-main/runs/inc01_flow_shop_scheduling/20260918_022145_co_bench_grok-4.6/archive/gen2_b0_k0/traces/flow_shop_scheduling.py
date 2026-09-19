def solve(**kwargs):
    n = kwargs['n']
    m = kwargs['m']
    matrix = kwargs['matrix']
    # Compute total processing time for each job (sum across all machines)
    totals = [sum(row) for row in matrix]
    # Sort jobs by total time ascending (shortest total processing time first heuristic)
    order = sorted(range(n), key=lambda i: totals[i])
    return {'job_sequence': [i + 1 for i in order]}