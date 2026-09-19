def flow_shop_local_search(matrix, seq):
    """Local search for flow shop via adjacent swaps.
    Runs a few passes of adjacent swap improvement. Each pass evaluates O(n) schedules;
    total time stays acceptable for n <= 500.

    Args:
        matrix: list[list[float]] — n_jobs x m_machines processing times (0-based)
        seq: list[int] — 0-based job sequence

    Returns:
        list[int] — improved 0-based job sequence
    """
    n = len(seq)
    m = len(matrix[0])

    def makespan(s):
        C = [[0.0] * m for _ in range(n)]
        for j in range(n):
            for k in range(m):
                if j == 0 and k == 0:
                    C[j][k] = matrix[s[j]][k]
                elif j == 0:
                    C[j][k] = C[j][k - 1] + matrix[s[j]][k]
                elif k == 0:
                    C[j][k] = C[j - 1][k] + matrix[s[j]][k]
                else:
                    C[j][k] = max(C[j - 1][k], C[j][k - 1]) + matrix[s[j]][k]
        return C[-1][-1]

    best_seq = seq[:]
    best_cm = makespan(best_seq)
    for _ in range(5):  # small number of passes
        improved = False
        for i in range(n - 1):
            temp = best_seq[:]
            temp[i], temp[i + 1] = temp[i + 1], temp[i]
            cm = makespan(temp)
            if cm < best_cm:
                best_seq = temp
                best_cm = cm
                improved = True
        if not improved:
            break
    return best_seq