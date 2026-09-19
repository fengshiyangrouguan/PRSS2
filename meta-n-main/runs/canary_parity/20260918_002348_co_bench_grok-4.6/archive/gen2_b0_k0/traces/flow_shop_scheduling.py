def solve(**kwargs):
    n = kwargs['n']
    m = kwargs['m']
    matrix = kwargs['matrix']
    p1 = [matrix[i][0] for i in range(n)]
    pm = [matrix[i][m-1] for i in range(n)]
    group1 = [i for i in range(n) if p1[i] <= pm[i]]
    group2 = [i for i in range(n) if p1[i] > pm[i]]
    group1_sorted = sorted(group1, key=lambda i: p1[i])
    group2_sorted = sorted(group2, key=lambda i: pm[i], reverse=True)
    job_sequence = group1_sorted + group2_sorted
    return {'job_sequence': [j + 1 for j in job_sequence]}