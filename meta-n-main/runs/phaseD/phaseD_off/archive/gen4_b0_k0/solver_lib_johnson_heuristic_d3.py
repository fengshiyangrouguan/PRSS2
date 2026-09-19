def johnson_heuristic(matrix: list[list[int]]) -> list[int]:
    """Constructive heuristic for flow-shop scheduling (Johnson's rule for m=2, total-time sort otherwise).

    Args:
        matrix: list[list[int]] where matrix[j][k] is processing time of job j on machine k (0-based)

    Returns:
        list[int]: job indices (0-based) in the constructed sequence order
    """
    n = len(matrix)
    m = len(matrix[0]) if matrix else 0
    if m == 2:
        jobs = list(range(n))
        a = [matrix[j][0] for j in jobs]
        b = [matrix[j][1] for j in jobs]
        A = [j for j in jobs if a[j] < b[j]]
        B = [j for j in jobs if a[j] > b[j]]
        A.sort(key=lambda j: a[j])
        B.sort(key=lambda j: b[j], reverse=True)
        order = A + B
    else:
        totals = [sum(row) for row in matrix]
        order = sorted(range(n), key=lambda i: totals[i])
    return order