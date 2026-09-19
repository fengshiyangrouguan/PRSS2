import itertools

def solve(**kwargs):
    n = kwargs['n']
    m = kwargs['m']
    matrix = kwargs['matrix']
    if n == 0:
        return {'job_sequence': []}

    def makespan(perm):
        C = [[0] * m for _ in range(n)]
        for i in range(n):
            for j in range(m):
                if i == 0 and j == 0:
                    C[i][j] = matrix[perm[i]][j]
                elif i == 0:
                    C[i][j] = C[i][j - 1] + matrix[perm[i]][j]
                elif j == 0:
                    C[i][j] = C[i - 1][j] + matrix[perm[i]][j]
                else:
                    C[i][j] = max(C[i - 1][j], C[i][j - 1]) + matrix[perm[i]][j]
        return C[n - 1][m - 1]

    if n <= 9:
        best_m = float('inf')
        best_perm = None
        for p in itertools.permutations(range(n)):
            mspan = makespan(p)
            if mspan < best_m:
                best_m = mspan
                best_perm = p
        perm = best_perm
    else:
        totals = [sum(row) for row in matrix]
        jobs = sorted(range(n), key=lambda j: totals[j], reverse=True)
        seq = [jobs[0], jobs[1]]
        for k in range(2, n):
            job = jobs[k]
            best_pos = 0
            best_m = float('inf')
            for pos in range(len(seq) + 1):
                new_seq = seq[:pos] + [job] + seq[pos:]
                mspan = makespan(new_seq)
                if mspan < best_m:
                    best_m = mspan
                    best_pos = pos
            seq = seq[:best_pos] + [job] + seq[best_pos:]
        perm = seq
    return {'job_sequence': [x + 1 for x in perm]}
