def solve(**kwargs):
    n = kwargs['n']
    m = kwargs['m']
    matrix = kwargs['matrix']
    jobs = list(range(n))
    A = [matrix[i][0] for i in range(n)]
    B = [matrix[i][m - 1] for i in range(n)]
    jobs.sort(key=lambda x: (A[x], -B[x]))
    return {'job_sequence': [j + 1 for j in jobs]}