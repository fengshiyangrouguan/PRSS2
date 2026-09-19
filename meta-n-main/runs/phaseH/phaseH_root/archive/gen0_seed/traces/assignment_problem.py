def solve(**kwargs):
    """
    Solves an instance of the Assignment Problem using the Hungarian algorithm.

    Args:
        n: int
        cost_matrix: square n x n matrix-like object

    Returns:
        dict with:
            "total_cost": minimal assignment cost
            "assignment": list of 1-indexed (item, agent) tuples
    """
    n = kwargs.get("n")
    cost_matrix = kwargs.get("cost_matrix")

    if n is None:
        n = len(cost_matrix)

    if n == 0:
        return {"total_cost": 0, "assignment": []}

    # Convert to a fast indexable Python list of lists without requiring numpy.
    try:
        cm = cost_matrix.tolist()
    except AttributeError:
        cm = [list(row) for row in cost_matrix]

    # Hungarian algorithm, 1-indexed internally.
    # u and v are potentials, p[j] is the row matched to column j.
    u = [0] * (n + 1)
    v = [0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)

    inf = float("inf")

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (n + 1)
        used = [False] * (n + 1)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = 0

            row = cm[i0 - 1]
            ui0 = u[i0]

            for j in range(1, n + 1):
                if not used[j]:
                    cur = row[j - 1] - ui0 - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j

            for j in range(0, n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        # Augment along the found path.
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    # Build row -> column assignment.
    assigned_col = [0] * (n + 1)
    for j in range(1, n + 1):
        assigned_col[p[j]] = j

    assignment = []
    total_cost = 0

    for i in range(1, n + 1):
        j = assigned_col[i]
        assignment.append((i, j))
        total_cost += cm[i - 1][j - 1]

    return {"total_cost": total_cost, "assignment": assignment}