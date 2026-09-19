def solve(**kwargs):
    """
    Solves an instance of the Assignment Problem.

    Input kwargs:
      - n: int
      - cost_matrix: 2D array-like with shape (n, n)

    Returns:
      dict with:
        - "total_cost": minimal/feasible total assignment cost
        - "assignment": list of 1-indexed (item, agent) tuples
    """
    import time

    n = int(kwargs.get("n"))
    cost_matrix = kwargs.get("cost_matrix")

    if n <= 0:
        return {"total_cost": 0, "assignment": []}

    def _clean_total(x):
        try:
            return x.item()
        except Exception:
            return x

    # Exact Hungarian algorithm for small/medium dense instances.
    def hungarian(cm, n):
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
                row = cm[i0 - 1]
                delta = INF
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

        row_to_col = [0] * (n + 1)
        for j in range(1, n + 1):
            if p[j] != 0:
                row_to_col[p[j]] = j

        assignment = []
        total = 0
        for i in range(1, n + 1):
            j = row_to_col[i]
            assignment.append((i, j))
            total += cm[i - 1][j - 1]

        return _clean_total(total), assignment

    # Fast feasible heuristic for large instances.
    def greedy_assignment(cm, n, time_limit=8.5):
        start = time.time()
        deadline = start + time_limit

        # Estimate row urgency by best-second-best gap.
        row_info = []
        for i in range(n):
            row = cm[i]
            best = None
            second = None
            bestj = -1

            for j in range(n):
                val = row[j]
                if best is None or val < best:
                    second = best
                    best = val
                    bestj = j
                elif second is None or val < second:
                    second = val

            if second is None:
                gap = 0
            else:
                gap = second - best
            row_info.append((-gap, best, i, bestj))

            if time.time() > deadline:
                break

        if len(row_info) < n:
            # Fallback if even preprocessing is too slow.
            used = [False] * n
            col_for_row = [-1] * n
            for i in range(n):
                row = cm[i]
                bestj = -1
                best = None
                for j in range(n):
                    if not used[j]:
                        val = row[j]
                        if best is None or val < best:
                            best = val
                            bestj = j
                if bestj < 0:
                    for j in range(n):
                        if not used[j]:
                            bestj = j
                            break
                used[bestj] = True
                col_for_row[i] = bestj
            assignment = [(i + 1, col_for_row[i] + 1) for i in range(n)]
            total = sum(cm[i][col_for_row[i]] for i in range(n))
            return _clean_total(total), assignment

        row_info.sort()
        used = [False] * n
        col_for_row = [-1] * n

        for _, _, i, preferred in row_info:
            if preferred >= 0 and not used[preferred]:
                col_for_row[i] = preferred
                used[preferred] = True
                continue

            row = cm[i]
            bestj = -1
            best = None
            for j in range(n):
                if not used[j]:
                    val = row[j]
                    if best is None or val < best:
                        best = val
                        bestj = j

            if bestj < 0:
                for j in range(n):
                    if not used[j]:
                        bestj = j
                        break

            col_for_row[i] = bestj
            used[bestj] = True

        # Limited 2-swap local improvement.
        # This is deliberately conservative to avoid timeouts.
        improved = True
        passes = 0
        while improved and passes < 2 and time.time() < deadline:
            improved = False
            passes += 1
            for a in range(n):
                if time.time() >= deadline:
                    break
                ca = col_for_row[a]
                rowa = cm[a]
                olda = rowa[ca]
                for b in range(a + 1, n):
                    cb = col_for_row[b]
                    rowb = cm[b]
                    old = olda + rowb[cb]
                    new = rowa[cb] + rowb[ca]
                    if new < old:
                        col_for_row[a], col_for_row[b] = cb, ca
                        improved = True
                        ca = cb
                        olda = rowa[ca]

        assignment = [(i + 1, col_for_row[i] + 1) for i in range(n)]
        total = sum(cm[i][col_for_row[i]] for i in range(n))
        return _clean_total(total), assignment

    # Threshold chosen to keep pure-Python exact method safely under the hard wall-clock limit.
    if n <= 400:
        total_cost, assignment = hungarian(cost_matrix, n)
    else:
        total_cost, assignment = greedy_assignment(cost_matrix, n, time_limit=8.5)

    return {"total_cost": total_cost, "assignment": assignment}