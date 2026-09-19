def auction_assignment(cost_matrix: list, n: int = None, time_limit: float = 8.5) -> tuple:
    """Compute a feasible assignment using a time-limited auction algorithm for minimization.

    Args:
        cost_matrix: list — square cost matrix supporting cost_matrix[i][j] indexing.
        n: int — problem dimension; if None, uses len(cost_matrix).
        time_limit: float — approximate wall-clock seconds available for the search.

    Returns:
        tuple — (total_cost, assignment), where total_cost is the summed selected cost and
        assignment is a list of 1-based (row, column) pairs assigning each row to one
        distinct column.
    """
    import time
    import math

    if n is None:
        n = len(cost_matrix)
    n = int(n)
    if n <= 0:
        return 0, []

    start = time.time()
    deadline = start + max(0.1, float(time_limit))

    def row_at(i: int):
        r = cost_matrix[i]
        try:
            return r.tolist()
        except AttributeError:
            return r

    # Scale epsilon to the observed cost range from a small sample.
    mn = None
    mx = None
    step = max(1, n // 32)
    for i in range(0, n, step):
        r = row_at(i)
        for j in range(0, n, step):
            v = r[j]
            if mn is None or v < mn:
                mn = v
            if mx is None or v > mx:
                mx = v
    if mn is None:
        mn, mx = 0, 1
    span = float(mx - mn) if mx != mn else 1.0

    # A moderate epsilon gives fast convergence; final small-epsilon passes improve quality.
    eps_values = [span / max(8.0, n ** 0.5), span / max(64.0, n), 1.0e-9]

    price = [0.0] * n
    owner = [-1] * n
    col_for_row = [-1] * n

    # Cheap initialization: assign each row to its current cheapest reduced-cost column if free.
    free_rows = []
    for i in range(n):
        if time.time() >= deadline:
            free_rows.append(i)
            continue
        r = row_at(i)
        best_j = 0
        best_v = r[0]
        for j in range(1, n):
            v = r[j]
            if v < best_v:
                best_v = v
                best_j = j
        if owner[best_j] < 0:
            owner[best_j] = i
            col_for_row[i] = best_j
        else:
            free_rows.append(i)

    # Any rows not assigned by initialization enter auction.
    for i in range(n):
        if col_for_row[i] < 0 and i not in free_rows:
            free_rows.append(i)

    for eps in eps_values:
        if time.time() >= deadline:
            break
        # Re-auction all rows at smaller epsilon except keep current assignment as a warm start.
        if eps != eps_values[0]:
            free_rows = []
            for i in range(n):
                if col_for_row[i] < 0:
                    free_rows.append(i)

        head = 0
        rounds_without_progress = 0
        while head < len(free_rows) and time.time() < deadline:
            i = free_rows[head]
            head += 1
            if col_for_row[i] >= 0:
                continue

            r = row_at(i)
            best_j = 0
            best = r[0] + price[0]
            second = float("inf")

            for j in range(1, n):
                val = r[j] + price[j]
                if val < best:
                    second = best
                    best = val
                    best_j = j
                elif val < second:
                    second = val

            if second == float("inf"):
                second = best
            bid = (second - best) + eps
            price[best_j] += bid

            old = owner[best_j]
            owner[best_j] = i
            col_for_row[i] = best_j
            if old >= 0:
                col_for_row[old] = -1
                free_rows.append(old)
                rounds_without_progress += 1
            else:
                rounds_without_progress = 0

            # Guard against pathological cycling under too-small epsilon.
            if rounds_without_progress > 4 * n and eps <= eps_values[-1]:
                break

    # Complete any remaining unassigned rows greedily over free columns to guarantee validity.
    free_cols = [j for j in range(n) if owner[j] < 0]
    if free_cols:
        for i in range(n):
            if col_for_row[i] >= 0:
                continue
            r = row_at(i)
            best_pos = 0
            best_j = free_cols[0]
            best_v = r[best_j]
            for pos in range(1, len(free_cols)):
                j = free_cols[pos]
                v = r[j]
                if v < best_v:
                    best_v = v
                    best_j = j
                    best_pos = pos
            col_for_row[i] = best_j
            owner[best_j] = i
            last = free_cols.pop()
            if best_pos < len(free_cols):
                free_cols[best_pos] = last

    # Brief pair-swap polishing, focused on expensive selected rows.
    if time.time() < deadline:
        rows = list(range(n))
        try:
            rows.sort(key=lambda i: row_at(i)[col_for_row[i]], reverse=True)
        except Exception:
            pass
        limit_outer = min(n, 900)
        limit_inner = min(n, 700)
        improved = True
        passes = 0
        while improved and passes < 2 and time.time() < deadline:
            improved = False
            passes += 1
            for a in range(limit_outer):
                if time.time() >= deadline:
                    break
                i = rows[a]
                ci = col_for_row[i]
                ri = row_at(i)
                old_i = ri[ci]
                for b in range(a + 1, min(n, a + 1 + limit_inner)):
                    k = rows[b]
                    ck = col_for_row[k]
                    rk = row_at(k)
                    old = old_i + rk[ck]
                    new = ri[ck] + rk[ci]
                    if new < old:
                        col_for_row[i], col_for_row[k] = ck, ci
                        improved = True
                        break
                if improved:
                    break

    total = 0
    for i in range(n):
        total += row_at(i)[col_for_row[i]]
    try:
        total = total.item()
    except AttributeError:
        pass

    return total, [(i + 1, col_for_row[i] + 1) for i in range(n)]