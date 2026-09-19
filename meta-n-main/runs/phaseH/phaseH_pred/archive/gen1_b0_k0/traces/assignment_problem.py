def solve(**kwargs):
    """
    Solves an instance of the Assignment Problem.

    Input kwargs:
      - n: int
      - cost_matrix: 2D list-like or numpy.ndarray with shape (n, n)

    Returns:
      dict with:
        - "total_cost": minimal/heuristic total assignment cost
        - "assignment": list of 1-indexed (item, agent) tuples
    """
    n = kwargs["n"]
    cost_matrix = kwargs["cost_matrix"]

    if n <= 0:
        return {"total_cost": 0, "assignment": []}

    def _py_scalar(x):
        try:
            return x.item()
        except AttributeError:
            return x

    def hungarian(cm, n):
        # 1-indexed Hungarian algorithm for square minimization.
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
                row = cm[i0 - 1]
                delta = inf
                j1 = 0

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

            while True:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
                if j0 == 0:
                    break

        row_to_col = [0] * n
        for j in range(1, n + 1):
            if p[j] != 0:
                row_to_col[p[j] - 1] = j - 1

        assignment = []
        total = 0
        for i, j in enumerate(row_to_col):
            assignment.append((i + 1, j + 1))
            total += _py_scalar(cm[i][j])

        return {"total_cost": _py_scalar(total), "assignment": assignment}

    def greedy_assignment(cm, n):
        # Regret-ordered greedy heuristic, O(n^2).
        row_info = []

        for i in range(n):
            row = cm[i]
            best_j = 0
            best = row[0]
            second = float("inf")

            for j in range(1, n):
                c = row[j]
                if c < best:
                    second = best
                    best = c
                    best_j = j
                elif c < second:
                    second = c

            regret = second - best
            row_info.append((regret, i, best_j))

        row_info.sort(reverse=True)

        used = [False] * n
        row_to_col = [-1] * n

        for _, i, best_j in row_info:
            if not used[best_j]:
                chosen = best_j
            else:
                row = cm[i]
                chosen = -1
                best_free_cost = None

                for j in range(n):
                    if not used[j]:
                        c = row[j]
                        if chosen < 0 or c < best_free_cost:
                            chosen = j
                            best_free_cost = c

            row_to_col[i] = chosen
            used[chosen] = True

        assignment = []
        total = 0
        for i, j in enumerate(row_to_col):
            assignment.append((i + 1, j + 1))
            total += _py_scalar(cm[i][j])

        return {"total_cost": _py_scalar(total), "assignment": assignment}

    # Exact for small/medium instances; fast feasible solution for large ones.
    if n <= 450:
        return hungarian(cost_matrix, n)
    return greedy_assignment(cost_matrix, n)