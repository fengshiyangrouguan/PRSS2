def solve(**kwargs):
    import time
    import random
    import bisect

    start_time = time.time()
    deadline = start_time + 8.7

    C = int(kwargs.get("bin_capacity", 0))
    items = list(kwargs.get("items", []))
    n = int(kwargs.get("num_items", len(items)))

    if n == 0:
        return {"num_bins": 0, "bins": []}

    total = sum(items)
    lower_bound = (total + C - 1) // C if C > 0 else n

    indices = list(range(n))

    def build_bfd(order):
        bins = []
        loads = []
        rem_sorted = []  # (remaining_capacity, bin_id)

        for i in order:
            s = items[i]
            pos = bisect.bisect_left(rem_sorted, (s, -1))
            if pos == len(rem_sorted):
                bid = len(bins)
                bins.append([i + 1])
                loads.append(s)
                if C - s >= 0:
                    bisect.insort(rem_sorted, (C - s, bid))
                else:
                    bisect.insort(rem_sorted, (C - s, bid))
            else:
                rem, bid = rem_sorted.pop(pos)
                bins[bid].append(i + 1)
                loads[bid] += s
                bisect.insort(rem_sorted, (rem - s, bid))
        return bins

    def build_ffd(order):
        bins = []
        rem = []
        for i in order:
            s = items[i]
            placed = False
            for b in range(len(bins)):
                if rem[b] >= s:
                    bins[b].append(i + 1)
                    rem[b] -= s
                    placed = True
                    break
            if not placed:
                bins.append([i + 1])
                rem.append(C - s)
        return bins

    def normalize_bins(bins):
        return [b for b in bins if b]

    def bin_load(b):
        return sum(items[i - 1] for i in b)

    def try_eliminate_bins(bins):
        bins = [list(b) for b in bins if b]

        improved = True
        while improved and time.time() < deadline:
            improved = False
            loads = [bin_load(b) for b in bins]
            order_bins = sorted(range(len(bins)), key=lambda x: loads[x])

            for target in order_bins:
                if time.time() >= deadline:
                    break
                if target >= len(bins) or not bins[target]:
                    continue

                target_items = sorted(bins[target], key=lambda x: items[x - 1], reverse=True)

                other_bins = []
                other_loads = []
                old_to_new = {}
                for j, b in enumerate(bins):
                    if j != target:
                        old_to_new[j] = len(other_bins)
                        other_bins.append(list(b))
                        other_loads.append(sum(items[x - 1] for x in b))

                rem_sorted = []
                for j, load in enumerate(other_loads):
                    bisect.insort(rem_sorted, (C - load, j))

                success = True
                for it in target_items:
                    s = items[it - 1]
                    pos = bisect.bisect_left(rem_sorted, (s, -1))
                    if pos == len(rem_sorted):
                        success = False
                        break
                    rem, bid = rem_sorted.pop(pos)
                    other_bins[bid].append(it)
                    other_loads[bid] += s
                    bisect.insort(rem_sorted, (rem - s, bid))

                if success:
                    bins = other_bins
                    improved = True
                    break

        return bins

    def valid(bins):
        seen = [0] * n
        for b in bins:
            load = 0
            for idx in b:
                if idx < 1 or idx > n:
                    return False
                seen[idx - 1] += 1
                load += items[idx - 1]
            if load > C:
                return False
        return all(x == 1 for x in seen)

    best_bins = None

    # Deterministic strong starts
    orders = []

    orders.append(sorted(indices, key=lambda i: items[i], reverse=True))
    orders.append(sorted(indices, key=lambda i: (items[i] % C if C else items[i]), reverse=True))
    orders.append(sorted(indices, key=lambda i: (items[i], i), reverse=True))

    # Pair-friendly order: large, then items close to complement.
    desc = sorted(indices, key=lambda i: items[i], reverse=True)
    asc = sorted(indices, key=lambda i: items[i])
    mixed = []
    l, r = 0, len(desc) - 1
    while l <= r:
        mixed.append(desc[l])
        l += 1
        if l <= r:
            mixed.append(desc[r])
            r -= 1
    orders.append(mixed)

    for order in orders:
        if time.time() >= deadline:
            break
        cand = build_bfd(order)
        cand = try_eliminate_bins(cand)
        if best_bins is None or len(cand) < len(best_bins):
            best_bins = cand
            if len(best_bins) <= lower_bound:
                break

        if time.time() >= deadline:
            break
        if n <= 3000:
            cand2 = build_ffd(order)
            cand2 = try_eliminate_bins(cand2)
            if len(cand2) < len(best_bins):
                best_bins = cand2
                if len(best_bins) <= lower_bound:
                    break

    # Randomized restarts with mild perturbation.
    if best_bins is None:
        best_bins = build_bfd(sorted(indices, key=lambda i: items[i], reverse=True))

    rng = random.Random(1234567 + n + total)
    attempts = 0
    max_attempts = 200 if n <= 400 else (60 if n <= 1500 else 12)

    while len(best_bins) > lower_bound and attempts < max_attempts and time.time() < deadline:
        attempts += 1

        if n <= 2500:
            # Shuffle within size classes / near-size classes.
            arr = indices[:]
            rng.shuffle(arr)
            if attempts % 3 == 0 and C > 0:
                # Slight preference for difficult large items but randomized.
                arr.sort(key=lambda i: (-(items[i] // max(1, C // 20 + 1)), rng.random()))
            else:
                arr.sort(key=lambda i: (-items[i], rng.random()))

            cand = build_bfd(arr)
            cand = try_eliminate_bins(cand)
            if len(cand) < len(best_bins):
                best_bins = cand
        else:
            break

    best_bins = normalize_bins(best_bins)

    # Final safety: if somehow invalid, fall back to simple singleton/FFD valid construction.
    if not valid(best_bins):
        fallback_order = sorted(indices, key=lambda i: items[i], reverse=True)
        best_bins = build_bfd(fallback_order)
        best_bins = normalize_bins(best_bins)

    return {
        "num_bins": len(best_bins),
        "bins": best_bins
    }