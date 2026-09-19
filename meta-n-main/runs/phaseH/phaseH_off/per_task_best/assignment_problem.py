def solve(**kwargs):
    """
    Solves an instance of the Assignment Problem.

    Input kwargs:
      - n: int
      - cost_matrix: 2D list-like / ndarray-like with shape (n, n)

    Returns:
      {"total_cost": minimal/feasible total cost, "assignment": [(item, agent), ...]}
    """
    import time

    n = int(kwargs["n"])
    cost_matrix = kwargs["cost_matrix"]

    if n <= 0:
        return {"total_cost": 0, "assignment": []}

    # Exact Hungarian algorithm for small / moderate instances.
    if n <= 450:
        # Convert once for faster repeated indexing.
        try:
            a0 = cost_matrix.tolist()
        except AttributeError:
            a0 = [list(row) for row in cost_matrix]

        # Hungarian algorithm for minimization, 1-indexed internal arrays.
        # Based on potentials, O(n^3).
        INF = float("inf")
        u = [0] * (n + 1)
        v = [0] * (n + 1)
        p = [0] * (n + 1)
        way = [0] * (n + 1)

        for i in range(1, n + 1):
            p[0] = i
            j0 = 0
            minv = [INF] * (n + 1)
            used = [False] * (n + 1)

            while True:
                used[j0] = True
                i0 = p[j0]
                delta = INF
                j1 = 0
                row = a0[i0 - 1]

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

        row_to_col = [0] * (n + 1)
        for j in range(1, n + 1):
            row_to_col[p[j]] = j

        assignment = []
        total_cost = 0
        for i in range(1, n + 1):
            j = row_to_col[i]
            assignment.append((i, j))
            total_cost += a0[i - 1][j - 1]

        return {"total_cost": total_cost, "assignment": assignment}

    # Large instances: return a valid feasible assignment quickly.
    # Greedy row-by-row minimum among unused columns, with a safety deadline.
    start = time.time()
    deadline = start + 8.5

    used_col = [False] * n
    assignment_cols = [-1] * n

    for i in range(n):
        if time.time() > deadline:
            break

        row = cost_matrix[i]
        best_j = -1
        best_val = None

        # Scan all columns and choose the cheapest unused column.
        for j in range(n):
            if not used_col[j]:
                val = row[j]
                if best_j < 0 or val < best_val:
                    best_val = val
                    best_j = j

        if best_j < 0:
            break

        assignment_cols[i] = best_j
        used_col[best_j] = True

    # Complete any unfinished rows with remaining columns to ensure feasibility.
    remaining_cols = [j for j in range(n) if not used_col[j]]
    ptr = 0
    for i in range(n):
        if assignment_cols[i] < 0:
            assignment_cols[i] = remaining_cols[ptr]
            ptr += 1

    assignment = []
    total_cost = 0
    for i, j in enumerate(assignment_cols):
        assignment.append((i + 1, j + 1))
        total_cost += cost_matrix[i][j]

    return {"total_cost": total_cost, "assignment": assignment}