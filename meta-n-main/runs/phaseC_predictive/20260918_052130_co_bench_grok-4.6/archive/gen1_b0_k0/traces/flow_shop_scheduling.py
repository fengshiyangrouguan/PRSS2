def solve(**kwargs):
    n = kwargs['n']
    m = kwargs['m']
    matrix = kwargs['matrix']
    if n <= 3:
        from itertools import permutations
        best_seq = None
        best_makespan = float('inf')
        for perm in permutations(range(n)):
            ms = compute_makespan(perm, matrix)
            if ms < best_makespan:
                best_makespan = ms
                best_seq = perm
        return {'job_sequence': [j + 1 for j in best_seq]}
    else:
        from itertools import permutations
        totals = [sum(row) for row in matrix]
        sorted_jobs = sorted(range(n), key=lambda j: totals[j], reverse=True)
        seed_jobs = sorted_jobs[:3]
        best_seed = None
        best_seed_ms = float('inf')
        for seed_perm in permutations(seed_jobs):
            ms = compute_makespan(seed_perm, matrix)
            if ms < best_seed_ms:
                best_seed_ms = ms
                best_seed = seed_perm
        current = list(best_seed)
        remaining = [j for j in sorted_jobs[3:]]
        for job in remaining:
            best_pos = 0
            best_ms = float('inf')
            for pos in range(len(current) + 1):
                temp = current[:pos] + [job] + current[pos:]
                ms = compute_makespan(temp, matrix)
                if ms < best_ms:
                    best_ms = ms
                    best_pos = pos
            current = current[:best_pos] + [job] + current[best_pos:]
        return {'job_sequence': [j + 1 for j in current]}

def compute_makespan(sequence, matrix):
    n = len(sequence)
    m = len(matrix[0])
    completion = [[0 for _ in range(m)] for _ in range(n)]
    for i in range(n):
        job = sequence[i]
        for j in range(m):
            if i == 0 and j == 0:
                completion[i][j] = matrix[job][j]
            elif i == 0:
                completion[i][j] = completion[i][j - 1] + matrix[job][j]
            elif j == 0:
                completion[i][j] = completion[i - 1][j] + matrix[job][j]
            else:
                completion[i][j] = max(completion[i - 1][j], completion[i][j - 1]) + matrix[job][j]
    return completion[n - 1][m - 1]