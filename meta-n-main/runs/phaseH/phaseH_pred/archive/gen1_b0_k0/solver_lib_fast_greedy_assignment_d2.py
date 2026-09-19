def fast_greedy_assignment(cost_matrix: list, n: int, time_limit: float = 8.5) -> dict:
    """Construct a fast feasible assignment using regret-ordered greedy selection.

    Args:
        cost_matrix: list — square n x n matrix-like object of numeric assignment costs.
        n: int — number of rows/jobs and columns/agents.
        time_limit: float — soft time budget in seconds for the heuristic.

    Returns:
        dict — {'total_cost': numeric, 'assignment': list[tuple[int, int]]}, where
        assignment contains 1-indexed (row, column) pairs forming a feasible permutation.
    """
    import time

    start = time.time()
    if n <= 0:
        return {"total_cost": 0, "assignment": []}

    try:
        cm = cost_matrix.tolist()
    except AttributeError:
        cm = cost_matrix

    # First pass: best column and regret for each row.
    row_info = []
    for i in range(n):
        row = cm[i]
        best_c = None
        best_j = 0
        second_c = None

        for j in range(n):
            c = row[j]
            if best_c is None or c < best_c:
                second_c = best_c
                best_c = c
                best_j = j
            elif second_c is None or c < second_c:
                second_c = c

        if second_c is None:
            regret = 0
        else:
            regret = second_c - best_c
        row_info.append((-regret, best_c, i, best_j))

    # High regret first; tie by low best cost.
    row_info.sort()

    free = [True] * n
    remaining = n
    assigned_col = [-1] * n

    for _, _, i, preferred_j in row_info:
        if remaining <= 0:
            break

        if free[preferred_j]:
            assigned_col[i] = preferred_j
            free[preferred_j] = False
            remaining -= 1
            continue

        row = cm[i]
        best_j = -1
        best_c = None

        # Full row scan is still only O(n^2) overall and gives much better cost
        # than taking an arbitrary remaining column.
        for j in range(n):
            if free[j]:
                c = row[j]
                if best_c is None or c < best_c:
                    best_c = c
                    best_j = j

        if best_j >= 0:
            assigned_col[i] = best_j
            free[best_j] = False
            remaining -= 1

        # If time is nearly exhausted, finish with arbitrary free columns.
        if time.time() - start > time_limit and remaining > 0:
            free_cols = [j for j in range(n) if free[j]]
            k = 0
            for ii in range(n):
                if assigned_col[ii] < 0:
                    assigned_col[ii] = free_cols[k]
                    k += 1
            break

    # Safety fill for any unassigned rows.
    if any(j < 0 for j in assigned_col):
        free_cols = [j for j in range(n) if free[j]]
        k = 0
        for i in range(n):
            if assigned_col[i] < 0:
                assigned_col[i] = free_cols[k]
                k += 1

    total_cost = 0
    assignment = []
    for i in range(n):
        j = assigned_col[i]
        assignment.append((i + 1, j + 1))
        total_cost += cm[i][j]

    return {"total_cost": total_cost, "assignment": assignment}