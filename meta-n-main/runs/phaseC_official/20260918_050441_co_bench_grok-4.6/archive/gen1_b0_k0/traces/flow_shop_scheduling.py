def solve(**kwargs):
    matrix = kwargs['matrix']
    n = kwargs['n']
    m = kwargs['m']
    if n == 0:
        return {'job_sequence': []}
    def makespan(seq, matrix):
        n = len(seq)
        m = len(matrix[0])
        C = [[0] * m for _ in range(n)]
        for j in range(n):
            for k in range(m):
                if k == 0:
                    C[j][k] = C[j-1][k] + matrix[seq[j]][k] if j > 0 else matrix[seq[j]][k]
                else:
                    C[j][k] = max(C[j-1][k], C[j][k-1]) + matrix[seq[j]][k]
        return C[n-1][m-1]
    def neh(matrix):
        n = len(matrix)
        if n == 0:
            return []
        totals = [sum(row) for row in matrix]
        jobs = sorted(range(n), key=lambda i: totals[i], reverse=True)
        if n == 1:
            return [jobs[0]]
        seq = [jobs[0], jobs[1]]
        if makespan([jobs[0], jobs[1]], matrix) > makespan([jobs[1], jobs[0]], matrix):
            seq = [jobs[1], jobs[0]]
        for i in range(2, n):
            job = jobs[i]
            best_seq = seq[:]
            best_makes = makespan(seq, matrix)
            for pos in range(len(seq) + 1):
                new_seq = seq[:pos] + [job] + seq[pos:]
                mks = makespan(new_seq, matrix)
                if mks < best_makes:
                    best_makes = mks
                    best_seq = new_seq
            seq = best_seq
        return seq
    def flow_shop_local_search(matrix, seq):
        n = len(seq)
        if n <= 1:
            return seq
        best_seq = seq[:]
        best_m = makespan(seq, matrix)
        improved = True
        while improved:
            improved = False
            best_delta = 0
            best_i = -1
            for i in range(n - 1):
                new_seq = seq[:]
                new_seq[i], new_seq[i + 1] = new_seq[i + 1], new_seq[i]
                m = makespan(new_seq, matrix)
                if m < best_m:
                    best_m = m
                    best_seq = new_seq[:]
                    best_i = i
                    best_delta = m - best_m
                    improved = True
            if best_i == -1:
                break
            seq = best_seq
        return seq
    seq = neh(matrix)
    seq = flow_shop_local_search(matrix, seq)
    return {'job_sequence': [x + 1 for x in seq]}