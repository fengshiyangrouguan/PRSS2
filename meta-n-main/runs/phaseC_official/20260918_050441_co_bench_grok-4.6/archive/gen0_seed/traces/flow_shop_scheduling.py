def solve(**kwargs):
    n = kwargs['n']
    m = kwargs['m']
    matrix = kwargs['matrix']
    if n == 0:
        return {'job_sequence': []}
    if n == 1:
        return {'job_sequence': [1]}
    if m == 1:
        return {'job_sequence': list(range(1, n + 1))}
    if m == 2:
        # Johnson's rule for m=2 (optimal)
        jobs = list(range(n))
        A = [i for i in jobs if matrix[i][0] < matrix[i][1]]
        B = [i for i in jobs if matrix[i][0] > matrix[i][1]]
        C = [i for i in jobs if matrix[i][0] == matrix[i][1]]
        A.sort(key=lambda i: matrix[i][0] - matrix[i][1])
        B.sort(key=lambda i: matrix[i][1] - matrix[i][0], reverse=True)
        seq = A + B + C
        return {'job_sequence': [s + 1 for s in seq]}
    else:
        # NEH heuristic for m > 2
        totals = [sum(row) for row in matrix]
        sorted_jobs = sorted(range(n), key=lambda i: totals[i], reverse=True)
        current = sorted_jobs[:2]
        def makespan(seq):
            p = [[matrix[j][k] for k in range(m)] for j in seq]
            C = [[0] * m for _ in range(len(seq))]
            for j in range(len(seq)):
                for k in range(m):
                    if j == 0 and k == 0:
                        C[j][k] = p[j][k]
                    elif j == 0:
                        C[j][k] = C[j][k - 1] + p[j][k]
                    elif k == 0:
                        C[j][k] = C[j - 1][k] + p[j][k]
                    else:
                        C[j][k] = max(C[j - 1][k], C[j][k - 1]) + p[j][k]
            return C[-1][-1]
        best_seq = current
        best_m = makespan(current)
        if makespan(current[::-1]) < best_m:
            best_seq = current[::-1]
            best_m = makespan(best_seq)
        remaining = [j for j in sorted_jobs[2:]]
        for job in remaining:
            best_pos = 0
            best_m_for_pos = float('inf')
            for pos in range(len(best_seq) + 1):
                temp_seq = best_seq[:pos] + [job] + best_seq[pos:]
                cm = makespan(temp_seq)
                if cm < best_m_for_pos:
                    best_m_for_pos = cm
                    best_pos = pos
            best_seq = best_seq[:best_pos] + [job] + best_seq[best_pos:]
        seq = [j + 1 for j in best_seq]
        return {'job_sequence': seq}