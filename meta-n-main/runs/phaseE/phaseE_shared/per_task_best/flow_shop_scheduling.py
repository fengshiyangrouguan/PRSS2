def solve(**kwargs):
    n = kwargs['n']
    m = kwargs['m']
    matrix = kwargs['matrix']
    if m == 2:
        jobs = list(range(n))
        a = [matrix[j][0] for j in jobs]
        b = [matrix[j][1] for j in jobs]
        A = [j for j in jobs if a[j] < b[j]]
        B = [j for j in jobs if a[j] > b[j]]
        A.sort(key=lambda j: a[j])
        B.sort(key=lambda j: b[j], reverse=True)
        order = A + B
        job_sequence = [j + 1 for j in order]
    else:
        totals = [sum(row) for row in matrix]
        order = sorted(range(n), key=lambda i: totals[i])
        job_sequence = [i + 1 for i in order]
    return {'job_sequence': job_sequence}