def solve_bin_packing(bin_capacity: float, items: list, time_limit: float = 8.5) -> tuple:
    """Solve 1D bin packing: pack item sizes into the fewest bins.

    Combines best-fit / first-fit decreasing, a greedy subset-sum bin-completion
    constructor, and a bin-elimination local search with a memoized, node-budgeted
    DFS reinsertion, plus randomized restarts.

    Args:
        bin_capacity: float — maximum total size that fits in one bin.
        items: list — item sizes; item i has size items[i] (0-based item ids).
        time_limit: float — approximate wall-clock second budget.

    Returns:
        tuple[int, list[list[int]]] — (num_bins, bins); bins is a list of lists of
        0-based item indices that partition range(len(items)) with no repeats.
    """
    import time
    import random
    import math
    import sys

    try:
        sys.setrecursionlimit(20000)
    except Exception:
        pass

    cap = bin_capacity
    n = len(items)
    if n == 0:
        return 0, []

    sizes = list(items)
    deadline = time.perf_counter() + max(0.5, float(time_limit))

    def _ff(order):
        bins, loads = [], []
        for i in order:
            s = sizes[i]
            placed = False
            for j in range(len(loads)):
                if loads[j] + s <= cap:
                    bins[j].append(i)
                    loads[j] += s
                    placed = True
                    break
            if not placed:
                bins.append([i])
                loads.append(s)
        return bins, loads

    def _bf(order):
        bins, loads = [], []
        for i in order:
            s = sizes[i]
            bj, br = -1, cap + 1
            for j in range(len(loads)):
                r = cap - loads[j]
                if s <= r < br:
                    br, bj = r, j
            if bj < 0:
                bins.append([i])
                loads.append(s)
            else:
                bins[bj].append(i)
                loads[bj] += s
        return bins, loads

    desc = sorted(range(n), key=lambda i: -sizes[i])

    b1, l1 = _ff(desc)
    b2, l2 = _bf(desc)
    if len(b1) <= len(b2):
        best_bins, best_loads = b1, l1
    else:
        best_bins, best_loads = b2, l2

    total = sum(sizes)
    lb = max(int(math.ceil(total / cap)), sum(1 for s in sizes if 2 * s > cap))

    # ---- subset-sum maximum fill of a single bin ----
    def _max_subset(vals, rem, dl, node_budget):
        m = len(vals)
        suf = [0] * (m + 1)
        for i in range(m - 1, -1, -1):
            suf[i] = suf[i + 1] + vals[i][0]
        best_sum = [0]
        best_sel = [[]]
        state = [0]

        def dfs(i, cur, left, sel):
            if state[0] > node_budget or time.perf_counter() > dl:
                return
            state[0] += 1
            if cur > best_sum[0]:
                best_sum[0] = cur
                best_sel[0] = sel[:]
                if cur == rem:
                    return
            if i >= m or cur + suf[i] <= best_sum[0]:
                return
            s = vals[i][0]
            if s <= left:
                sel.append(vals[i][1])
                dfs(i + 1, cur + s, left - s, sel)
                sel.pop()
                if best_sum[0] == rem:
                    return
            dfs(i + 1, cur, left, sel)

        dfs(0, 0, rem, [])
        return best_sel[0]

    # ---- greedy bin-completion constructor ----
    def _greedy_completion(dl, node_budget=6000):
        remaining = desc[:]
        bins, loads = [], []
        while remaining:
            if time.perf_counter() > dl:
                tb, tl = _ff(remaining)
                bins.extend(tb)
                loads.extend(tl)
                break
            seed = remaining.pop(0)
            s = sizes[seed]
            rem_cap = cap - s
            if rem_cap <= 0 or not remaining:
                bins.append([seed])
                loads.append(s)
                continue
            vals = sorted(((sizes[i], i) for i in remaining), reverse=True)
            chosen = _max_subset(vals, rem_cap, dl, node_budget)
            bins.append([seed] + chosen)
            loads.append(s + sum(sizes[i] for i in chosen))
            if chosen:
                cset = set(chosen)
                remaining = [i for i in remaining if i not in cset]
        return bins, loads

    # ---- DFS reinsertion of removed items into existing bins ----
    def _reinsert(place, bins, loads, dl, node_budget=60000):
        place = sorted(place, reverse=True)
        m = len(place)
        if m == 0:
            return True
        if not loads:
            return False
        total_rem = sum(cap - x for x in loads)
        if sum(p[0] for p in place) > total_rem:
            return False
        if place[0][0] > max(cap - x for x in loads):
            return False

        suf = [0] * (m + 1)
        for i in range(m - 1, -1, -1):
            suf[i] = suf[i + 1] + place[i][0]

        B = len(loads)
        memo_on = B <= 30
        failed = set()
        state = [0]
        tref = [total_rem]

        def dfs(i):
            if i == m:
                return True
            if state[0] > node_budget or time.perf_counter() > dl:
                return False
            state[0] += 1
            if suf[i] > tref[0]:
                return False
            key = None
            if memo_on:
                key = (i, tuple(sorted(loads)))
                if key in failed:
                    return False
            s, idx = place[i]
            cand = []
            for j in range(B):
                r = cap - loads[j]
                if r >= s:
                    cand.append((r, j))
            if not cand:
                if key is not None:
                    failed.add(key)
                return False
            cand.sort()
            tried = set()
            for _, j in cand:
                if loads[j] in tried:
                    continue
                tried.add(loads[j])
                loads[j] += s
                bins[j].append(idx)
                tref[0] -= s
                if dfs(i + 1):
                    return True
                tref[0] += s
                bins[j].pop()
                loads[j] -= s
            if key is not None:
                failed.add(key)
            return False

        return dfs(0)

    # ---- try to remove one or two bins ----
    def _try_reduce():
        nonlocal best_bins, best_loads
        B = len(best_bins)
        if B <= lb:
            return False
        order = sorted(range(B), key=lambda j: best_loads[j])
        for j in order:
            if time.perf_counter() > deadline - 0.02:
                return False
            place = [(sizes[i], i) for i in best_bins[j]]
            nb = [best_bins[k][:] for k in range(B) if k != j]
            nl = [best_loads[k] for k in range(B) if k != j]
            dl = min(deadline, time.perf_counter() + max(0.15, (deadline - time.perf_counter()) * 0.4))
            if _reinsert(place, nb, nl, dl):
                best_bins, best_loads = nb, nl
                return True

        if B > 2:
            cand = sorted(range(B), key=lambda j: best_loads[j])[:14]
            for ai in range(len(cand)):
                if time.perf_counter() > deadline - 0.02:
                    return False
                for bi in range(ai + 1, len(cand)):
                    if time.perf_counter() > deadline - 0.02:
                        return False
                    ja, jb = cand[ai], cand[bi]
                    place = ([(sizes[i], i) for i in best_bins[ja]]
                             + [(sizes[i], i) for i in best_bins[jb]])
                    if len(place) > 45:
                        continue
                    rem_idx = [k for k in range(B) if k != ja and k != jb]
                    nb = [best_bins[k][:] for k in rem_idx]
                    nl = [best_loads[k] for k in rem_idx]
                    if sum(p[0] for p in place) > sum(cap - x for x in nl):
                        continue
                    dl = min(deadline, time.perf_counter() + max(0.2, (deadline - time.perf_counter()) * 0.5))
                    if _reinsert(place, nb, nl, dl):
                        best_bins, best_loads = nb, nl
                        return True
        return False

    def _random_restart():
        nonlocal best_bins, best_loads
        order = desc[:]
        for _ in range(max(1, n // 8)):
            a = random.randrange(n)
            b = random.randrange(n)
            order[a], order[b] = order[b], order[a]
        rb, rl = _ff(order)
        rb2, rl2 = _bf(order)
        if len(rb2) < len(rb):
            rb, rl = rb2, rl2
        if len(rb) < len(best_bins):
            best_bins, best_loads = rb, rl
            return True
        return False

    # ---- attempt a better start via subset-sum bin completion ----
    if len(best_bins) > lb and n <= 250 and time.perf_counter() < deadline - 0.5:
        dl_c = min(deadline, time.perf_counter() + 1.5)
        try:
            cb, cl = _greedy_completion(dl_c)
            if len(cb) < len(best_bins):
                best_bins, best_loads = cb, cl
        except RecursionError:
            pass

    # ---- main search loop ----
    stall = 0
    while time.perf_counter() < deadline - 0.05 and len(best_bins) > lb and stall < 80:
        if _try_reduce():
            stall = 0
            continue
        if _random_restart():
            stall = 0
            continue
        stall += 1

    return len(best_bins), [list(b) for b in best_bins]