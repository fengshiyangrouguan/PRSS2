import time
import itertools

def solve(**kwargs):
    n = kwargs['n']
    m = kwargs['m']
    matrix = kwargs['matrix']
    if n == 0:
        return {'job_sequence': []}
    start_time = time.time()
    best_makespan = float('inf')
    best_seq = None
    for perm in itertools.permutations(range(n)):
        elapsed = time.time() - start_time
        if elapsed > 9.0:  # safety margin
            break
        ms = compute_makespan(perm, matrix)
        if ms < best_makespan:
            best_makespan = ms
            best_seq = perm
    if best_seq is None:
        best_seq = tuple(range(n))  # fallback
    return {'job_sequence': [j + 1 for j in best_seq]}

def compute_makespan(order, matrix):
    n = len(order)
    m = len(matrix[0])
    C = [[0] * m for _ in range(n)]
    for j in range(m):
        for i in range(n):
            job = order[i]
            if i == 0 and j == 0:
                C[i][j] = matrix[job][j]
            elif i == 0:
                C[i][j] = C[i][j - 1] + matrix[job][j]
            elif j == 0:
                C[i][j] = C[i - 1][j] + matrix[job][j]
            else:
                C[i][j] = max(C[i - 1][j], C[i][j - 1]) + matrix[job][j]
    return C[n - 1][m - 1]