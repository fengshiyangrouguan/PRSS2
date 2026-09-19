def solve(**kwargs):
    """
    Solve the one-dimensional bin packing problem for a single test case.
    Returns {'num_bins': int, 'bins': [[1-based indices], ...]}.
    """
    import time
    import random
    import sys

    start = time.perf_counter()
    deadline = start + 8.0  # keep a safe margin under the 10 s hard limit

    C = int(kwargs['bin_capacity'])
    items = list(kwargs['items'])
    n = int(kwargs.get('num_items', len(items)))
    if n == 0:
        return {'num_bins': 0, 'bins': []}
    if len(items) > n:
        items = items[:n]
    elif len(items) < n:
        n = len(items)

    sys.setrecursionlimit(max(1000, n * 2 + 100))

    # item tuples: (size, 1-based original index)
    indexed = [(items[i], i + 1) for i in range(n)]
    indexed.sort(key=lambda x: -x[0])
    sorted_sizes = [s for s, _ in indexed]
    sorted_ids = [idx for _, idx in indexed]

    total = sum(sorted_sizes)
    lb = max((total + C - 1) // C, max(sorted_sizes))

    # ----- helpers -------------------------------------------------
    def first_fit(order):
        bins = []
        loads = []
        for s, idx in order:
            placed = False
            # linear first-fit: usually breaks fast
            for i in range(len(bins)):
                if loads[i] + s <= C:
                    bins[i].append(idx)
                    loads[i] += s
                    placed = True
                    break
            if not placed:
                bins.append([idx])
                loads.append(s)
        return bins, loads

    def pack_valid(bins):
        seen = set()
        for b in bins:
            load = 0
            for idx in b:
                if idx in seen or idx < 1 or idx > n:
                    return False
                seen.add(idx)
                load += items[idx - 1]
                if load > C:
                    return False
        return len(seen) == n

    def empty_bin_elimination(bins, loads):
        """Try to remove bins by relocating all of their items."""
        improved = True
        while improved and time.perf_counter() < deadline:
            improved = False
            src_order = sorted(range(len(bins)), key=lambda x: loads[x])
            for src in src_order:
                if time.perf_counter() > deadline:
                    break
                if src >= len(bins):
                    continue
                src_items = bins[src]
                if not src_items:
                    bins.pop(src)
                    loads.pop(src)
                    improved = True
                    break

                targets = [j for j in range(len(bins)) if j != src]
                target_caps = [loads[j] for j in targets]

                # greedy best-fit relocation
                placements = []
                ok = True
                for idx in sorted(src_items, key=lambda x: -items[x - 1]):
                    s = items[idx - 1]
                    best_t = -1
                    best_res = C + 1
                    for ti in range(len(targets)):
                        if target_caps[ti] + s <= C:
                            res = C - (target_caps[ti] + s)
                            if res < best_res:
                                best_res = res
                                best_t = ti
                    if best_t == -1:
                        ok = False
                        break
                    target_caps[best_t] += s
                    placements.append((idx, targets[best_t]))

                if ok:
                    for idx, t in placements:
                        bins[t].append(idx)
                        loads[t] += items[idx - 1]
                    bins.pop(src)
                    loads.pop(src)
                    improved = True
                    break

                # fallback DFS for bins with few items
                if len(src_items) <= 10:
                    order_t = sorted(range(len(targets)), key=lambda x: loads[targets[x]])
                    targets2 = [targets[t] for t in order_t]
                    caps2 = [loads[j] for j in targets2]
                    it_sorted = sorted(src_items, key=lambda x: -items[x - 1])
                    sizes2 = [items[idx - 1] for idx in it_sorted]
                    assign = [-1] * len(sizes2)

                    def dfs_place(pos):
                        if pos == len(sizes2):
                            return True
                        s = sizes2[pos]
                        seen_cap = set()
                        for i in range(len(caps2)):
                            c = caps2[i]
                            if c in seen_cap:
                                continue
                            if c + s <= C:
                                caps2[i] += s
                                assign[pos] = i
                                if dfs_place(pos + 1):
                                    return True
                                caps2[i] -= s
                            seen_cap.add(c)
                        return False

                    if dfs_place(0):
                        for k, ti in enumerate(assign):
                            g = targets2[ti]
                            bins[g].append(it_sorted[k])
                            loads[g] += sizes2[k]
                        bins.pop(src)
                        loads.pop(src)
                        improved = True
                        break
        return bins, loads

    # ----- initial FFD solution ------------------------------------
    base_order = list(indexed)
    best_bins, best_loads = first_fit(base_order)
    best = len(best_bins)

    if best == lb:
        return {'num_bins': best, 'bins': best_bins}

    # ----- exact branch-and-bound for small instances --------------
    if n <= 45:
        exact_deadline = min(deadline, start + 5.0)

        suffix = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix[i] = suffix[i + 1] + sorted_sizes[i]

        LB_all = max((total + C - 1) // C,
                     sum(1 for s in sorted_sizes if s > C // 2))

        bb_loads = [0] * n
        bb_bins = [[] for _ in range(n)]

        class Timeout(Exception):
            pass

        def run_exact():
            nonlocal best, best_bins
            nodes = 0

            def dfs(level, k):
                nonlocal best, best_bins, nodes
                nodes += 1
                if nodes & 0xFFF == 0:
                    if time.perf_counter() > exact_deadline:
                        raise Timeout
                if level == n:
                    if k < best:
                        best = k
                        best_bins = [list(bb_bins[i]) for i in range(k)]
                    return
                if max(k, LB_all) >= best:
                    return

                s = sorted_sizes[level]
                seen_loads = set()
                placed = False
                for i in range(k):
                    L = bb_loads[i]
                    if L + s <= C and L not in seen_loads:
                        bb_bins[i].append(sorted_ids[level])
                        bb_loads[i] = L + s
                        dfs(level + 1, k)
                        bb_loads[i] = L
                        bb_bins[i].pop()
                        seen_loads.add(L)
                        placed = True
                if not placed:
                    if k + 1 < best:
                        bb_bins[k].append(sorted_ids[level])
                        bb_loads[k] = s
                        dfs(level + 1, k + 1)
                        bb_loads[k] = 0
                        bb_bins[k].pop()

            try:
                dfs(0, 0)
            except Timeout:
                pass

        run_exact()

        if best == lb:
            return {'num_bins': best, 'bins': best_bins}

    # ----- local improvement on the incumbent ----------------------
    best_loads = [sum(items[idx - 1] for idx in b) for b in best_bins]
    best_bins, best_loads = empty_bin_elimination(best_bins, best_loads)
    best = len(best_bins)
    if best == lb:
        return {'num_bins': best, 'bins': best_bins}

    # ----- randomized restart search -------------------------------
    max_size = max(sorted_sizes)
    min_size = min(sorted_sizes)
    span = max_size - min_size
    # deterministic but instance-specific seed
    random.seed(str(kwargs.get('id', '')) + str(n))

    while time.perf_counter() < deadline:
        order = list(base_order)
        if span > 0:
            order.sort(key=lambda x: (-x[0] + random.random() * span * 0.5))
        else:
            random.shuffle(order)

        # a few random swaps to increase diversity
        swaps = random.randrange(0, max(1, n // 20))
        for _ in range(swaps):
            i = random.randrange(n)
            j = random.randrange(n)
            order[i], order[j] = order[j], order[i]

        bins, loads = first_fit(order)
        if len(bins) < best:
            bins, loads = empty_bin_elimination(bins, loads)
            if len(bins) < best:
                best = len(bins)
                best_bins = bins
                best_loads = loads
                if best == lb:
                    break

    # safety check: if somehow invalid, fall back to FFD
    if not pack_valid(best_bins):
        best_bins, _ = first_fit(base_order)

    return {'num_bins': len(best_bins), 'bins': best_bins}