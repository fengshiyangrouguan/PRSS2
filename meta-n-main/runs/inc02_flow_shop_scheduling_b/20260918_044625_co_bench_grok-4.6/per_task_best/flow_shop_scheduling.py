def solve(**kwargs):
    n = kwargs['n']
    m = kwargs['m']
    matrix = kwargs['matrix']
    def compute_makespan(perm):
        C = [[0] * m for _ in range(n)]
        for i in range(n):
            job = perm[i]
            for j in range(m):
                if i == 0:
                    C[i][j] = matrix[job][j] if j == 0 else C[i][j-1] + matrix[job][j]
                else:
                    C[i][j] = max(C[i-1][j], C[i][j-1]) + matrix[job][j]
        return C[n-1][m-1]
    if n <= 9:
        best_makespan = float('inf')
        best_seq = None
        for perm in itertools.permutations(range(n)):
            makes = compute_makespan(perm)
            if makes < best_makespan:
                best_makespan = makes
                best_seq = perm
        return {'job_sequence': [j+1 for j in best_seq]}
    else:
        # For larger n, return a reasonable heuristic permutation (e.g., sorted by total processing time)
        total_times = [sum(row) for row in matrix]
        sorted_jobs = sorted(range(n), key=lambda j: total_times[j], reverse=True)
        return {'job_sequence': [j+1 for j in sorted_jobs]}