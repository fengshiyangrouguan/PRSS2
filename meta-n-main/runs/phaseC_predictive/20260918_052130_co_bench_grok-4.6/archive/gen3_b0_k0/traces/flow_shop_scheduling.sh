```python
def solve(**kwargs):
    n = kwargs['n']
    m = kwargs['m']
    matrix = kwargs['matrix']
    if n <= 3:
        from itertools import permutations
        best_seq = None
        best_m = float('inf')
        for p in permutations(range(n)):
            mspan = 0.0
            C = [[0.0] * m for _ in range(n)]
            for j in range(n):
                job = p[j]
                for i in range(m):
                    if j == 0 and i == 0:
                        C[j][i] = matrix[job][i]
                    elif j == 0:
                        C[j][i] = C[j][i-1] + matrix[job][i]
                    elif i == 0:
                        C[j][i] = C[j-1][i] + matrix[job][i]
                    else:
                        C[j][i] = max(C[j-1][i], C[j][i-1]) + matrix[job][i]
            mspan = C[n-1][m-1]
            if mspan < best_m:
                best_m = mspan
                best_seq = p
        return {'job_sequence': [j+1 for j in best_seq]}
    # n > 3: improved NEH
    totals = [sum(matrix[j]) for j in range(n)]
    sorted_jobs = sorted(range(n), key=lambda j: totals[j], reverse=True)
    seed = sorted_jobs[:3]
    from itertools import permutations
    best_seed_m = float('inf')
    best_seed = None
    for seed_perm in permutations(seed):
        C = [[0.0] * m for _ in range(3)]
        for j in range(3):
            job = seed_perm[j]
            for i in range(m):
                if j == 0 and i == 0:
                    C[j][i] = matrix[job][i]
                elif j == 0:
                    C[j][i] = C[j][i-1] + matrix[job][i]
                elif i == 0:
                    C[j][i] = C[j-1][i] + matrix[job][i]
                else:
                    C[j][i] = max(C[j-1][i], C[j][i-1]) + matrix[job][i]
        mspan = C[2][m-1]
        if mspan < best_seed_m:
            best_seed_m = mspan
            best_seed = seed_perm
    current_seq = list(best_seed)
    remaining = [j for j in sorted_jobs[3:]]
    for job in remaining:
        best_pos_m = float('inf')
        best_pos = -1
        for pos in range(len(current_seq) + 1):
            new_seq = current_seq[:pos] + [job] + current_seq[pos:]
            C = [[0.0] * m for _ in range(len(new_seq))]
            for j in range(len(new_seq)):
                jjob = new_seq[j]
                for i in range(m):
                    if j == 0 and i == 0:
                        C[j][i] = matrix[jjob][i]
                    elif j == 0:
                        C[j][i] = C[j][i-1] + matrix[jjob][i]
                    elif i == 0:
                        C[j][i] = C[j-1][i] + matrix[jjob][i]
                    else:
                        C[j][i] = max(C[j-1][i], C[j][i-1]) + matrix[jjob][i]
            mspan = C[len(new_seq)-1][m-1]
            if mspan < best_pos_m:
                best_pos_m = mspan
                best_pos = pos
        current_seq = current_seq[:best_pos] + [job] + current_seq[best_pos:]
    # one adjacent-swap local search pass
    best_m = float('inf')
    for j in range(len(current_seq)):
        job = current_seq[j]
        for i in range(m):
            if j == 0 and i == 0:
                best_m = matrix[job][i]
            elif j == 0:
                best_m = best_m + matrix[job][i]
            elif i == 0:
                best_m = best_m + matrix[job][i]
            else:
                best_m = max(best_m, best_m) + matrix[job][i]  # placeholder; actually recompute full
    # correct local search
    best_m = float('inf')
    for j in range(len(current_seq)):
        job = current_seq[j]
        for i in range(m):
            if j == 0 and i == 0:
                best_m = matrix[job][i]
            elif j