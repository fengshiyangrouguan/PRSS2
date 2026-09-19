def solve(**kwargs):
    import time
    import random

    n = kwargs["n"]
    cost_matrix = kwargs["cost_matrix"]

    if n <= 0:
        return {"total_cost": 0, "assignment": []}

    def getc(i, j):
        return cost_matrix[i][j]

    # Exact Hungarian algorithm for small/medium instances.
    if n <= 450:
        a = cost_matrix
        u = [0] * (n + 1)
        v = [0] * (n + 1)
        p = [0] * (n + 1)
        way = [0] * (n + 1)

        for i in range(1, n + 1):
            p[0] = i
            j0 = 0
            minv = [float("inf")] * (n + 1)
            used = [False] * (n + 1)

            while True:
                used[j0] = True
                i0 = p[j0]
                delta = float("inf")
                j1 = 0

                row = a[i0 - 1]
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

        row_to_col = [0] * n
        for j in range(1, n + 1):
            if p[j] != 0:
                row_to_col[p[j] - 1] = j - 1

        total_cost = 0
        assignment = []
        for i, j in enumerate(row_to_col):
            total_cost += getc(i, j)
            assignment.append((i + 1, j + 1))

        return {"total_cost": total_cost, "assignment": assignment}

    # Large-instance heuristic:
    # 1. Regret-ordered greedy initialization.
    # 2. Time-bounded pairwise swap local improvement.
    start = time.time()
    deadline = start + 8.5

    row_info = []
    for i in range(n):
        row = cost_matrix[i]
        best_j = -1
        best = float("inf")
        second = float("inf")

        for j in range(n):
            c = row[j]
            if c < best:
                second = best
                best = c
                best_j = j
            elif c < second:
                second = c

        regret = second - best if second != float("inf") else 0
        row_info.append((-regret, best, i, best_j))

    row_info.sort()

    assigned_col = [-1] * n
    col_used = [False] * n

    for _, _, i, preferred in row_info:
        if preferred >= 0 and not col_used[preferred]:
            assigned_col[i] = preferred
            col_used[preferred] = True
        else:
            row = cost_matrix[i]
            best_j = -1
            best = float("inf")
            for j in range(n):
                if not col_used[j]:
                    c = row[j]
                    if c < best:
                        best = c
                        best_j = j
            assigned_col[i] = best_j
            col_used[best_j] = True

        if time.time() > deadline:
            break

    # If deadline was hit during greedy, complete any missing assignment quickly.
    if any(j < 0 for j in assigned_col):
        free_cols = [j for j in range(n) if not col_used[j]]
        k = 0
        for i in range(n):
            if assigned_col[i] < 0:
                assigned_col[i] = free_cols[k]
                k += 1

    total_cost = 0
    for i, j in enumerate(assigned_col):
        total_cost += getc(i, j)

    # Deterministic adjacent/randomized 2-swap improvement under deadline.
    # A swap of rows i,k is accepted if:
    # c[i][col_k] + c[k][col_i] < c[i][col_i] + c[k][col_k]
    rng = random.Random(1234567)

    # First pass over structured pairs, cheap and often useful.
    i = 0
    while i + 1 < n and time.time() < deadline:
        j = i + 1
        ci = assigned_col[i]
        cj = assigned_col[j]
        old = getc(i, ci) + getc(j, cj)
        new = getc(i, cj) + getc(j, ci)
        if new < old:
            assigned_col[i], assigned_col[j] = cj, ci
            total_cost += new - old
        i += 2

    # Random pair swaps until the time budget is nearly exhausted.
    checks = 0
    while time.time() < deadline:
        i = rng.randrange(n)
        k = rng.randrange(n)
        if i == k:
            continue

        ci = assigned_col[i]
        ck = assigned_col[k]

        old = getc(i, ci) + getc(k, ck)
        new = getc(i, ck) + getc(k, ci)

        if new < old:
            assigned_col[i], assigned_col[k] = ck, ci
            total_cost += new - old

        checks += 1
        if checks % 2048 == 0 and time.time() >= deadline:
            break

    assignment = [(i + 1, assigned_col[i] + 1) for i in range(n)]
    return {"total_cost": total_cost, "assignment": assignment}