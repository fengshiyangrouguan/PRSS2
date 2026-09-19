def solve(**kwargs):
    n = kwargs.get('n', 0)
    m = kwargs.get('m', 0)
    matrix = kwargs.get('matrix', [])
    if n <= 1:
        return {'job_sequence': list(range(1, n + 1))}
    if m == 2:
        jobs = list(range(n))
        a = [matrix[j][0] for j in jobs]
        b = [matrix[j][1] for j in jobs]
        group1 = [j for j in jobs if a[j] < b[j]]
        group2 = [j for j in jobs if a[j] > b[j]]
        group1.sort(key=lambda j: a[j])
        group2.sort(key=lambda j: b[j], reverse=True)
        seq = group1 + group2
    else:
        jobs = list(range(n))
        jobs.sort(key=lambda j: sum(matrix[j]))
        seq = jobs
    return {'job_sequence': [j + 1 for j in seq]}