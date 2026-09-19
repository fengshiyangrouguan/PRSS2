import math

def solve(**kwargs):
    n = kwargs['n']
    m = kwargs['m']
    matrix = kwargs['matrix']
    if n == 0:
        return {'job_sequence': []}
    if n == 1:
        return {'job_sequence': [1]}
    # Compute total processing time for each job
    totals = [sum(row) for row in matrix]
    jobs = list(range(n))
    # Sort jobs by total time descending
    jobs.sort(key=lambda j: totals[j], reverse=True)
    # Calculate makespan for a sequence
    def calc_makespan(seq):
        times = [[matrix[j][i] for i in range(m)] for j in seq]
        C = [[0] * m for _ in range(n)]
        for j in range(n):
            for i in range(m):
                if j == 0 and i == 0:
                    C[j][i] = times[j][i]
                elif j == 0:
                    C[j][i] = C[j][i - 1] + times[j][i]
                elif i == 0:
                    C[j][i] = C[j - 1][i] + times[j][i]
                else:
                    C[j][i] = max(C[j - 1][i], C[j][i - 1]) + times[j][i]
        return C[-1][-1]
    # NEH heuristic
    if n == 2:
        seq = jobs[:]
    else:
        seq = [jobs[0], jobs[1]]
        for k in range(2, n):
            next_job = jobs[k]
            best_makespan = math.inf
            best_pos = 0
            for pos in range(len(seq) + 1):
                temp_seq = seq[:pos] + [next_job] + seq[pos:]
                ms = calc_makespan(temp_seq)
                if ms < best_makespan:
                    best_makespan = ms
                    best_pos = pos
            seq = seq[:best_pos] + [next_job] + seq[best_pos:]
    # Return 1-indexed permutation
    return {'job_sequence': [j + 1 for j in seq]}