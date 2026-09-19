def solve(**kwargs):
    n = kwargs['n']
    m = kwargs['m']
    matrix = kwargs['matrix']

    def compute_makespan(sequence, matrix):
        n = len(sequence)
        m = len(matrix[0])
        completion = [[0] * m for _ in range(n)]
        for i in range(n):
            job = sequence[i]
            for j in range(m):
                if i == 0 and j == 0:
                    completion[i][j] = matrix[job][j]
                elif i == 0:
                    completion[i][j] = completion[i][j-1] + matrix[job][j]
                elif j == 0:
                    completion[i][j] = completion[i-1][j] + matrix[job][j]
                else:
                    completion[i][j] = max(completion[i-1][j], completion[i][j-1]) + matrix[job][j]
        return completion[-1][-1]

    def neh_heuristic(matrix):
        n = len(matrix)
        m = len(matrix[0])
        total_times = [(sum(row), i) for i, row in enumerate(matrix)]
        total_times.sort(reverse=True)
        sequence = [total_times[0][1]]
        for k in range(1, n):
            job = total_times[k][1]
            best_makespan = float('inf')
            best_pos = 0
            for pos in range(len(sequence) + 1):
                temp_seq = sequence[:pos] + [job] + sequence[pos:]
                makespan = compute_makespan(temp_seq, matrix)
                if makespan < best_makespan:
                    best_makespan = makespan
                    best_pos = pos
            sequence.insert(best_pos, job)
        return sequence

    if n <= 5:
        best_sequence = None
        best_makespan = float('inf')
        for seq in itertools.permutations(range(n)):
            makespan = compute_makespan(seq, matrix)
            if makespan < best_makespan:
                best_makespan = makespan
                best_sequence = seq
        return {'job_sequence': [x + 1 for x in best_sequence]}
    else:
        best_sequence = neh_heuristic(matrix)
        return {'job_sequence': [x + 1 for x in best_sequence]}