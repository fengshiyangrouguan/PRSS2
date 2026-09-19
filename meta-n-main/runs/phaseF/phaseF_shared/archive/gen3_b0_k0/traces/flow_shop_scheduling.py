import itertools

def solve(**kwargs):
    n = kwargs['n']
    m = kwargs['m']
    matrix = kwargs['matrix']
    if n == 0:
        return {'job_sequence': []}
    best_makespan = float('inf')
    best_seq = None
    for seq in itertools.permutations(range(n)):
        C = [[0] * m for _ in range(n)]
        for i in range(n):
            for j in range(m):
                if i == 0 and j == 0:
                    C[i][j] = matrix[seq[i]][j]
                elif i == 0:
                    C[i][j] = C[i][j-1] + matrix[seq[i]][j]
                elif j == 0:
                    C[i][j] = C[i-1][j] + matrix[seq[i]][j]
                else:
                    C[i][j] = max(C[i-1][j], C[i][j-1]) + matrix[seq[i]][j]
        makespan = C[n-1][m-1]
        if makespan < best_makespan:
            best_makespan = makespan
            best_seq = seq
    job_sequence = [s + 1 for s in best_seq]
    return {'job_sequence': job_sequence}