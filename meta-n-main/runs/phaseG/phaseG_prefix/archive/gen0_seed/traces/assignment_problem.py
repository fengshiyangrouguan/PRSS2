def solve(**kwargs):
    """
    Solves an instance of the Assignment Problem using the Hungarian algorithm.
    """
    n = kwargs.get("n")
    cost_matrix = kwargs.get("cost_matrix")

    if n is None:
        n = len(cost_matrix)

    if n == 0:
        return {"total_cost": 0, "assignment": []}

    try:
        a = cost_matrix.tolist()
    except AttributeError:
        a = [list(row) for row in cost_matrix]

    # Hungarian algorithm for minimization, 1-indexed internally.
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
            row = a[i0 - 1]

            delta = inf
            j1 = 0

            for j in range(1, n + 1):
                if not used[j]:
                    cur = row[j - 1] - u[i0] - v[j]
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

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    item_to_agent = [0] * (n + 1)
    for j in range(1, n + 1):
        if p[j] != 0:
            item_to_agent[p[j]] = j

    assignment = []
    total_cost = 0
    for i in range(1, n + 1):
        j = item_to_agent[i]
        assignment.append((i, j))
        total_cost += a[i - 1][j - 1]

    try:
        total_cost = total_cost.item()
    except AttributeError:
        pass

    return {"total_cost": total_cost, "assignment": assignment}