def fast_greedy_assignment(cost_matrix: list, n: int = None, time_limit: float = 8.5) -> tuple:
    """Construct a fast feasible assignment for large dense assignment instances.

    Args:
        cost_matrix: list[list[number]] — square n by n cost matrix, or an object with row indexing.
        n: int — problem dimension; if None, uses len(cost_matrix).
        time_limit: float — approximate seconds available for construction and local improvement.

    Returns:
        tuple[float, list[tuple[int, int]]] — (total_cost, assignment), where assignment is
        a list of 1-based (row, column) pairs assigning every row to exactly one distinct column.
    """
    import time

    start = time.time()
    deadline = start + max(0.1, float(time_limit))
    if n is None:
        n = len(cost_matrix)
    n = int(n)

    if n <= 0:
        return 0, []

    def get_row(i):
        row = cost_matrix[i]
        try:
            return row.tolist()
        except AttributeError:
            return row

    # For each row, find best and second-best columns.  Rows with high regret are assigned first.
    row_info = []
    for i in range(n):
        row = get_row(i)
        best_c = 0
        best_v = row[0]
        second_v = None
        for j in range(1, n):
            v = row[j]
            if v < best_v:
                second_v = best_v
                best_v = v
                best_c = j
            elif second_v is None or v < second_v:
                second_v = v
        if second_v is None:
            second_v = best_v
        row_info.append((second_v - best_v, best_v, i, best_c))

    row_info.sort(reverse=True)

    free_cols = list(range(n))
    assigned_col = [-1] * n
    total = 0

    # Greedy regret assignment: each difficult row gets its cheapest remaining column.
    for _, __, i, pref in row_info:
        if time.time() > deadline:
            break
        row = get_row(i)
        best_pos = 0
        best_j = free_cols[0]
        best_v = row[best_j]

        # Fast path if preferred column is still free.
        # Linear membership would be too costly, so scan once anyway.
        for pos, j in enumerate(free_cols):
            v = row[j]
            if v < best_v:
                best_v = v
                best_j = j
                best_pos = pos

        assigned_col[i] = best_j
        total += best_v
        last = free_cols.pop()
        if best_pos < len(free_cols):
            free_cols[best_pos] = last

    # If time expired mid-construction, finish arbitrarily but validly.
    if free_cols:
        k = 0
        for i in range(n):
            if assigned_col[i] < 0:
                j = free_cols[k]
                assigned_col[i] = j
                total += get_row(i)[j]
                k += 1

    # Time-limited 2-row swap improvement.  This is cheap and often improves greedy collisions.
    # Try rows with largest selected costs first.
    if time.time() < deadline:
        rows = list(range(n))
        try:
            rows.sort(key=lambda i: get_row(i)[assigned_col[i]], reverse=True)
        except Exception:
            pass

        max_outer = n if n <= 1200 else min(n, 700)
        max_inner = n if n <= 1200 else min(n, 900)

        improved = True
        passes = 0
        while improved and passes < 2 and time.time() < deadline:
            improved = False
            passes += 1
            for ai in range(max_outer):
                if time.time() >= deadline:
                    break
                i = rows[ai]
                ci = assigned_col[i]
                row_i = get_row(i)
                old_i = row_i[ci]
                for bj in range(ai + 1, min(len(rows), ai + 1 + max_inner)):
                    k = rows[bj]
                    ck = assigned_col[k]
                    if ci == ck:
                        continue
                    row_k = get_row(k)
                    old = old_i + row_k[ck]
                    new = row_i[ck] + row_k[ci]
                    if new < old:
                        assigned_col[i], assigned_col[k] = ck, ci
                        total += new - old
                        improved = True
                        break
                if improved:
                    break

    assignment = [(i + 1, assigned_col[i] + 1) for i in range(n)]
    try:
        total = total.item()
    except AttributeError:
        pass
    return total, assignment