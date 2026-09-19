def solve(**kwargs):
    import time
    import bisect
    import random
    import math

    start_time = time.time()
    deadline = start_time + 9.2

    capacity = int(kwargs.get("bin_capacity"))
    items = list(kwargs.get("items", []))
    n = int(kwargs.get("num_items", len(items)))

    if n == 0:
        return {"num_bins": 0, "bins": []}

    # If an impossible item appears, still return a complete assignment.
    # Such cases are normally not present in valid bin-packing benchmarks.
    if any(x > capacity for x in items):
        return {"num_bins": n, "bins": [[i + 1] for i in range(n)]}

    total_size = sum(items)
    lower_bound = (total_size + capacity - 1) // capacity

    def pack_order(order):
        """Best-fit packing for a given order of 0-based item indices."""
        bins = []
        rems = []
        sorted_rems = []  # (remaining_capacity, bin_id)

        for idx in order:
            w = items[idx]
            pos = bisect.bisect_left(sorted_rems, (w, -1))
            if pos == len(sorted_rems):
                bid = len(bins)
                bins.append([idx + 1])
                rem = capacity - w
                rems.append(rem)
                bisect.insort(sorted_rems, (rem, bid))
            else:
                rem, bid = sorted_rems.pop(pos)
                bins[bid].append(idx + 1)
                rem -= w
                rems[bid] = rem
                bisect.insort(sorted_rems, (rem, bid))
        return bins

    # Build several strong deterministic initial solutions.
    indexed = list(range(n))
    decreasing = sorted(indexed, key=lambda i: (-items[i], i))
    increasing = sorted(indexed, key=lambda i: (items[i], i))

    best_bins = pack_order(decreasing)

    if len(best_bins) > lower_bound and time.time() < deadline:
        cand = pack_order(indexed)
        if len(cand) < len(best_bins):
            best_bins = cand

    if len(best_bins) > lower_bound and time.time() < deadline:
        cand = pack_order(increasing)
        if len(cand) < len(best_bins):
            best_bins = cand

    # Try a few randomized tie-breakings among similarly sized items.
    if len(best_bins) > lower_bound and n <= 5000:
        rng = random.Random(1234567 + n + capacity)
        attempts = 8 if n <= 1000 else 3
        for _ in range(attempts):
            if time.time() >= deadline:
                break
            order = decreasing[:]
            # Shuffle inside coarse size buckets to diversify while preserving decreasing structure.
            grouped = []
            i = 0
            while i < n:
                j = i + 1
                key = items[order[i]]
                while j < n and items[order[j]] == key:
                    j += 1
                block = order[i:j]
                rng.shuffle(block)
                grouped.extend(block)
                i = j
            cand = pack_order(grouped)
            if len(cand) < len(best_bins):
                best_bins = cand
                if len(best_bins) == lower_bound:
                    break

    def normalize_bins(bins):
        return [b for b in bins if b]

    def improve_by_elimination(bins):
        """Repeatedly try to remove one bin by repacking its items into the others."""
        bins = [list(b) for b in bins if b]
        changed = True
        while changed and len(bins) > lower_bound and time.time() < deadline:
            changed = False
            loads = [sum(items[i - 1] for i in b) for b in bins]
            rems = [capacity - x for x in loads]

            # Prefer eliminating sparsely loaded bins, but also try bins with few items first.
            candidates = sorted(range(len(bins)), key=lambda k: (loads[k], len(bins[k])))

            for kill in candidates:
                if time.time() >= deadline:
                    break

                moving = sorted(bins[kill], key=lambda x: -items[x - 1])
                temp_bins = [list(bins[i]) for i in range(len(bins)) if i != kill]
                temp_rems = [rems[i] for i in range(len(bins)) if i != kill]

                feasible = True
                for item_idx in moving:
                    w = items[item_idx - 1]
                    best_j = -1
                    best_after = capacity + 1
                    for j, r in enumerate(temp_rems):
                        if r >= w:
                            after = r - w
                            if after < best_after:
                                best_after = after
                                best_j = j
                                if after == 0:
                                    break
                    if best_j == -1:
                        feasible = False
                        break
                    temp_bins[best_j].append(item_idx)
                    temp_rems[best_j] -= w

                if feasible:
                    bins = temp_bins
                    changed = True
                    break
        return normalize_bins(bins)

    if len(best_bins) > lower_bound:
        cand = improve_by_elimination(best_bins)
        if len(cand) < len(best_bins):
            best_bins = cand

    # Exact branch-and-bound / feasibility search for small and medium instances.
    # It attempts to prove feasibility with k bins for k < current best.
    if len(best_bins) > lower_bound and n <= 70 and time.time() < deadline:
        sorted_items = sorted([(items[i], i + 1) for i in range(n)], reverse=True)
        weights = [x[0] for x in sorted_items]
        ids = [x[1] for x in sorted_items]

        suffix_sum = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_sum[i] = suffix_sum[i + 1] + weights[i]

        node_counter = [0]

        def feasible_with_k(k):
            rem = [capacity] * k
            assign = [[] for _ in range(k)]

            def dfs(pos):
                node_counter[0] += 1
                if node_counter[0] & 2047 == 0:
                    if time.time() >= deadline:
                        raise TimeoutError

                if pos == n:
                    return True

                # Remaining total capacity pruning.
                if suffix_sum[pos] > sum(rem):
                    return False

                w = weights[pos]
                item_id = ids[pos]

                seen_remainders = set()
                # Try tighter bins first.
                order_bins = sorted(range(k), key=lambda b: rem[b])
                for b in order_bins:
                    r = rem[b]
                    if r < w or r in seen_remainders:
                        continue
                    seen_remainders.add(r)

                    rem[b] -= w
                    assign[b].append(item_id)

                    if dfs(pos + 1):
                        return True

                    assign[b].pop()
                    rem[b] += w

                    # Symmetry: if placing into an empty bin failed, other empty bins are equivalent.
                    if r == capacity:
                        break

                return False

            try:
                if dfs(0):
                    return [list(b) for b in assign if b]
            except TimeoutError:
                return None
            return None

        target = len(best_bins) - 1
        k = lower_bound
        while k <= target and time.time() < deadline:
            result = feasible_with_k(k)
            if result is not None:
                best_bins = result
                target = k - 1
                break
            k += 1

    # Final validity safeguard: ensure every item appears exactly once.
    seen = set()
    valid = True
    for b in best_bins:
        load = 0
        for idx in b:
            if idx < 1 or idx > n or idx in seen:
                valid = False
                break
            seen.add(idx)
            load += items[idx - 1]
        if load > capacity:
            valid = False
        if not valid:
            break

    if not valid or len(seen) != n:
        best_bins = pack_order(decreasing)

    return {
        "num_bins": len(best_bins),
        "bins": best_bins
    }