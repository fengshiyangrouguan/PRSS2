def solve(**kwargs):
    import time
    import random
    import itertools

    start = time.perf_counter()
    deadline = start + 9.2

    cap = kwargs.get('bin_capacity')
    items = kwargs.get('items', [])
    n = len(items)

    if n == 0:
        return {'num_bins': 0, 'bins': []}

    # Sort items descending by size
    sorted_idx = sorted(range(n), key=lambda i: items[i], reverse=True)
    sorted_items = [(items[i], i) for i in sorted_idx]

    def ffd(order):
        bins = []
        loads = []
        for idx in order:
            s = items[idx]
            placed = False
            for j in range(len(bins)):
                if loads[j] + s <= cap:
                    bins[j].append(idx)
                    loads[j] += s
                    placed = True
                    break
            if not placed:
                bins.append([idx])
                loads.append(s)
        return bins, loads

    def bfd(order):
        bins = []
        loads = []
        for idx in order:
            s = items[idx]
            best_j = -1
            best_rem = cap + 1
            for j in range(len(bins)):
                rem = cap - loads[j]
                if rem >= s and rem < best_rem:
                    best_rem = rem
                    best_j = j
            if best_j == -1:
                bins.append([idx])
                loads.append(s)
            else:
                bins[best_j].append(idx)
                loads[best_j] += s
        return bins, loads

    # Initial solutions
    sol1, load1 = ffd(sorted_idx)
    sol2, load2 = bfd(sorted_idx)
    if len(sol1) <= len(sol2):
        bins, loads = sol1, load1
    else:
        bins, loads = sol2, load2

    # Lower bound
    total_size = sum(items)
    lb_sum = (total_size + cap - 1) // cap
    lb_half = sum(1 for s in items if 2 * s > cap)
    lower = max(lb_sum, lb_half)

    def try_pack_into_bins(place_items, bins, loads, cap, deadline):
        m = len(place_items)
        if m == 0:
            return True
        total_size = sum(s for s, _ in place_items)
        total_rem = sum(cap - l for l in loads)
        if total_size > total_rem:
            return False

        place_items.sort(key=lambda x: x[0], reverse=True)

        # Greedy fallback for many items
        if m > 25:
            added = []
            for s, idx in place_items:
                best_j = -1
                best_rem = cap + 1
                for j in range(len(loads)):
                    rem = cap - loads[j]
                    if rem >= s and rem < best_rem:
                        best_rem = rem
                        best_j = j
                if best_j == -1:
                    for bj, i, sz in reversed(added):
                        bins[bj].pop()
                        loads[bj] -= sz
                    return False
                bins[best_j].append(idx)
                loads[best_j] += s
                added.append((best_j, idx, s))
            return True

        suffix_sum = [0] * (m + 1)
        for i in range(m - 1, -1, -1):
            suffix_sum[i] = suffix_sum[i + 1] + place_items[i][0]

        failed = set()

        def dfs(i):
            if i == m:
                return True
            if time.perf_counter() > deadline:
                return False
            if suffix_sum[i] > sum(cap - l for l in loads):
                return False
            if place_items[i][0] > max(cap - l for l in loads):
                return False

            state = (i, tuple(sorted(loads)))
            if state in failed:
                return False

            s, idx = place_items[i]
            tried_loads = set()
            candidates = []
            for j in range(len(loads)):
                rem = cap - loads[j]
                if rem >= s:
                    candidates.append((rem, j))
            candidates.sort()

            for rem, j in candidates:
                if loads[j] in tried_loads:
                    continue
                tried_loads.add(loads[j])
                loads[j] += s
                bins[j].append(idx)
                if dfs(i + 1):
                    return True
                bins[j].pop()
                loads[j] -= s

            failed.add(state)
            return False

        return dfs(0)

    def remove_one(bins, loads, cap, deadline):
        B = len(bins)
        if B <= 1:
            return False
        order = sorted(range(B), key=lambda j: len(bins[j]))
        for j in order:
            if time.perf_counter() > deadline:
                return False
            new_bins = [bins[k][:] for k in range(B) if k != j]
            new_loads = [loads[k] for k in range(B) if k != j]
            place_items = [(items[idx], idx) for idx in bins[j]]
            if try_pack_into_bins(place_items, new_bins, new_loads, cap, deadline):
                bins[:] = new_bins
                loads[:] = new_loads
                return True
        return False

    def remove_set(bins, loads, cap, deadline, max_remove=2):
        B = len(bins)
        if B <= max_remove:
            return False
        candidates = sorted(range(B), key=lambda j: len(bins[j]))
        if B > 20:
            candidates = candidates[:15]

        for r in range(2, max_remove + 1):
            cnt = 0
            for combo in itertools.combinations(candidates, r):
                if time.perf_counter() > deadline:
                    return False
                total_items = sum(len(bins[j]) for j in combo)
                if total_items > 30:
                    continue
                total_size = sum(loads[j] for j in combo)
                combo_set = set(combo)
                rem_cap = sum(cap - loads[k] for k in range(B) if k not in combo_set)
                if total_size > rem_cap:
                    continue

                new_bins = [bins[k][:] for k in range(B) if k not in combo_set]
                new_loads = [loads[k] for k in range(B) if k not in combo_set]
                place_items = []
                for j in combo:
                    for idx in bins[j]:
                        place_items.append((items[idx], idx))

                if try_pack_into_bins(place_items, new_bins, new_loads, cap, deadline):
                    bins[:] = new_bins
                    loads[:] = new_loads
                    return True

                cnt += 1
                if cnt > 200:
                    break
        return False

    def pack_fixed_k(items_sorted, K, cap, deadline):
        n = len(items_sorted)
        if K == 0:
            return [] if n == 0 else None

        bins = [[] for _ in range(K)]
        loads = [0] * K
        suffix_sum = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_sum[i] = suffix_sum[i + 1] + items_sorted[i][0]

        failed = set()

        def dfs(i, used_bins):
            if i == n:
                return [b for b in bins if b]
            if time.perf_counter() > deadline:
                return None

            total_rem = sum(cap - loads[j] for j in range(used_bins)) + (K - used_bins) * cap
            if suffix_sum[i] > total_rem:
                return None

            state = (i, tuple(sorted(loads[:used_bins])))
            if state in failed:
                return None

            s, idx = items_sorted[i]
            tried_loads = set()
            candidates = []
            for j in range(used_bins):
                if loads[j] + s <= cap:
                    candidates.append((cap - loads[j], j))
            candidates.sort()

            for rem, j in candidates:
                if loads[j] in tried_loads:
                    continue
                tried_loads.add(loads[j])
                loads[j] += s
                bins[j].append(idx)
                res = dfs(i + 1, used_bins)
                if res is not None:
                    return res
                bins[j].pop()
                loads[j] -= s

            if used_bins < K and 0 not in tried_loads:
                bins[used_bins].append(idx)
                loads[used_bins] = s
                res = dfs(i + 1, used_bins + 1)
                if res is not None:
                    return res
                bins[used_bins].pop()
                loads[used_bins] = 0

            failed.add(state)
            return None

        return dfs(0, 0)

    # Main improvement loop
    while time.perf_counter() < deadline - 0.1 and len(bins) > lower:
        improved = False

        if remove_one(bins, loads, cap, deadline):
            improved = True
            continue

        if remove_set(bins, loads, cap, deadline, max_remove=2):
            improved = True
            continue

        if remove_set(bins, loads, cap, deadline, max_remove=3):
            improved = True
            continue

        if (n <= 60 or len(bins) <= 20) and len(bins) > lower:
            slice_deadline = min(deadline, time.perf_counter() + 2.0)
            res = pack_fixed_k(sorted_items, len(bins) - 1, cap, slice_deadline)
            if res is not None:
                bins = res
                loads = [sum(items[i] for i in b) for b in bins]
                improved = True
                continue

        if n > 1 and time.perf_counter() < deadline - 0.5:
            best_count = len(bins)
            best_bins = bins
            best_loads = loads
            iters = 100 if n > 500 else 500 if n > 200 else 2000
            for _ in range(iters):
                if time.perf_counter() > deadline - 0.5:
                    break
                order = sorted_idx[:]
                swaps = max(1, n // 10)
                for __ in range(swaps):
                    i = random.randrange(n - 1)
                    order[i], order[i + 1] = order[i + 1], order[i]
                new_bins, new_loads = ffd(order)
                if len(new_bins) < best_count:
                    best_count = len(new_bins)
                    best_bins = new_bins
                    best_loads = new_loads
            if best_count < len(bins):
                bins = best_bins
                loads = best_loads
                improved = True
                continue

        if not improved:
            break

    # Convert to 1-based indices
    final_bins = []
    for b in bins:
        final_bins.append([idx + 1 for idx in b])

    return {
        'num_bins': len(final_bins),
        'bins': final_bins
    }