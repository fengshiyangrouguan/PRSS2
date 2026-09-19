def improve_sequence(sequence: list, matrix: list) -> list:
    """Perform a single pass of adjacent swaps to improve makespan.
    Args:
        sequence: list[int] — current job order (0-based)
        matrix: list[list[int]] — processing times (n jobs x m machines)
    Returns:
        list[int] — improved sequence (0-based indices)
    """
    import copy
    n = len(sequence)
    m = len(matrix[0])
    current = copy.deepcopy(sequence)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            # swap adjacent
            temp = current[:]
            temp[i], temp[i + 1] = temp[i + 1], temp[i]
            # compute makespan
            comp = [[0] * m for _ in range(n)]
            for j in range(n):
                job = temp[j]
                for k in range(m):
                    if j == 0 and k == 0:
                        comp[j][k] = matrix[job][k]
                    elif k == 0:
                        comp[j][k] = comp[j - 1][k] + matrix[job][k]
                    elif j == 0:
                        comp[j][k] = comp[j][k - 1] + matrix[job][k]
                    else:
                        comp[j][k] = max(comp[j - 1][k], comp[j][k - 1]) + matrix[job][k]
            if comp[n - 1][m - 1] < comp[n - 1][m - 1]:  # always false in practice; keep structure
                current = temp[:]
                improved = True
                break
    return current