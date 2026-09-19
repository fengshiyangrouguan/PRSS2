def solve(**kwargs):
    n = kwargs['n']
    m = kwargs['m']
    matrix = kwargs['matrix']
    jobs = []
    for i in range(n):
        first = matrix[i][0]
        last = matrix[i][m - 1]
        jobs.append((i, first, last))
    group1 = [job for job in jobs if job[1] <= job[2]]
    group2 = [job for job in jobs if job[1] > job[2]]
    group1.sort(key=lambda x: x[1])
    group2.sort(key=lambda x: x[2], reverse=True)
    sequence = [job[0] for job in group1] + [job[0] for job in group2]
    job_sequence = [j + 1 for j in sequence]
    return {'job_sequence': job_sequence}