def solve(**kwargs):
    n = kwargs['n']
    m = kwargs['m']
    matrix = kwargs['matrix']
    if n == 0:
        return {'job_sequence': []}
    totals = [sum(row) for row in matrix]
    jobs = list(range(n))
    jobs.sort(key=lambda j: totals[j], reverse=True)
    def makespan(seq):
        if not seq:
            return 0
        C = [[0] * m for _ in range(n)]
        for k in range(m):
            C[0][k] = sum(matrix[seq[0]][:k + 1])
        for i in range(1, len(seq)):
            C[i][0] = C[i - 1][0] + matrix[seq[i]][0]
            for k in range(1, m):
                C[i][k] = max(C[i - 1][k], C[i][k - 1]) + matrix[seq[i]][k]
        return C[-1][-1]
    if n <= 2:
        best_seq = jobs
    else:
        best_seq = jobs[:2]
        best_ms = makespan(best_seq)
        for i in range(2, n):
            current_best = best_ms
            current_best_seq = best_seq[:]
            for pos in range(len(best_seq) + 1):
                temp_seq = best_seq[:pos] + [jobs[i]] + best_seq[pos:]
                ms = makespan(temp_seq)
                if ms < current_best:
                    current_best = ms
                    current_best_seq = temp_seq
            best_seq = current_best_seq
            best_ms = current_best
    return {'job_sequence': [j + 1 for j in best_seq]}