def solve(**kwargs):
    n = kwargs['n']
    m = kwargs['m']
    matrix = kwargs['matrix']
    if n == 0:
        return {'job_sequence': []}
    if n == 1:
        return {'job_sequence': [1]}

    def compute_makespan(seq):
        C = [[0] * m for _ in range(n)]
        C[0][0] = matrix[seq[0]][0]
        for k in range(1, m):
            C[0][k] = C[0][k - 1] + matrix[seq[0]][k]
        for i in range(1, n):
            C[i][0] = C[i - 1][0] + matrix[seq[i]][0]
            for k in range(1, m):
                C[i][k] = max(C[i - 1][k], C[i][k - 1]) + matrix[seq[i]][k]
        return C[n - 1][m - 1]

    # Compute total processing times
    totals = [sum(row) for row in matrix]
    jobs = list(range(n))
    jobs.sort(key=lambda j: totals[j], reverse=True)

    # Start with first two jobs
    seq = [jobs[0]]
    best_makespan = float('inf')
    best_seq = None
    for i in range(2):
        s = [jobs[1], jobs[0]] if i == 0 else [jobs[0], jobs[1]]
        ms = compute_makespan(s)
        if ms < best_makespan:
            best_makespan = ms
            best_seq = s[:]
    seq = best_seq

    # Insert remaining jobs
    remaining = [j for j in jobs[2:]]
    for j in remaining:
        best_makespan = float('inf')
        best_pos = 0
        for pos in range(len(seq) + 1):
            s = seq[:pos] + [j] + seq[pos:]
            ms = compute_makespan(s)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        seq = seq[:best_pos] + [j] + seq[best_pos:]

    return {'job_sequence': [x + 1 for x in seq]}