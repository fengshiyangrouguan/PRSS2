import itertools

def solve(**kwargs):
    n = kwargs['n']
    m = kwargs['m']
    matrix = kwargs['matrix']
    if n == 0:
        return {'job_sequence': []}
    if n <= 9:
        best_makespan = float('inf')
        best_perm = None
        for p in itertools.permutations(range(n)):
            ms = compute_makespan(p, matrix)
            if ms < best_makespan:
                best_makespan = ms
                best_perm = p
        job_sequence = [x + 1 for x in best_perm]
    else:
        job_sequence = neh_heuristic(matrix)
    return {'job_sequence': job_sequence}

def compute_makespan(perm, matrix):
    n = len(perm)
    m = len(matrix[0])
    C = [[0] * m for _ in range(n)]
    # First job
    C[0][0] = matrix[perm[0]][0]
    for j in range(1, m):
        C[0][j] = C[0][j - 1] + matrix[perm[0]][j]
    # Remaining jobs
    for i in range(1, n):
        C[i][0] = C[i - 1][0] + matrix[perm[i]][0]
        for j in range(1, m):
            C[i][j] = max(C[i - 1][j], C[i][j - 1]) + matrix[perm[i]][j]
    return C[n - 1][m - 1]

def neh_heuristic(matrix):
    n = len(matrix)
    m = len(matrix[0])
    # Compute total processing times
    totals = [sum(row) for row in matrix]
    # Sort jobs by total time descending
    jobs = sorted(range(n), key=lambda x: totals[x], reverse=True)
    if n == 1:
        return [1]
    # Start with first two jobs
    seq = [jobs[0], jobs[1]]
    # Insert remaining jobs
    for k in range(2, n):
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(seq) + 1):
            temp_seq = seq[:pos] + [jobs[k]] + seq[pos:]
            ms = compute_makespan(temp_seq, matrix)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq = seq[:best_pos] + [jobs[k]] + seq[best_pos:]
    return [x + 1 for x in seq]