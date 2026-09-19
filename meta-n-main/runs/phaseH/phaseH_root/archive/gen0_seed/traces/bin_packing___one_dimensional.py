def solve(**kwargs):
    import time
    import bisect
    import random

    start_time = time.time()
    deadline = start_time + 9.2

    capacity = int(kwargs.get("bin_capacity", 0))
    items = list(kwargs.get("items", []))
    n = int(kwargs.get("num_items", len(items)))

    if n == 0:
        return {"num_bins": 0, "bins": []}

    indexed = [(items[i], i + 1) for i in range(n)]

    # If malformed/impossible oversized items exist, the best valid-ish fallback is singleton bins.
    # Standard instances should not contain these.
    if capacity <= 0:
        return {"num_bins": n, "bins": [[i + 1] for i in range(n)]}

    total = sum(items)
    max_item = max(items) if items else 0
    lower_bound = max((total + capacity - 1) // capacity, 1 if n else 0)

    if max_item > capacity:
        return {"num_bins": n, "bins": [[i + 1] for i in range(n)]}

    if total <= capacity:
        return {"num_bins": 1, "bins": [list(range(1, n + 1))]}

    def best_fit_decreasing(order):
        bins = []
        loads = []
        # sorted list of (remaining_capacity, bin_id)
        residuals = []
        alive_bins = 0

        for size, idx in order:
            pos = bisect.bisect_left(residuals, (size, -1))
            if pos == len(residuals):
                bid = alive_bins
                alive_bins += 1
                bins.append([idx])
                loads.append(size)
                rem = capacity - size
                bisect.insort(residuals, (rem, bid))
            else:
                rem, bid = residuals.pop(pos)
                bins[bid].append(idx)
                loads[bid] += size
                rem -= size
                bisect.insort(residuals, (rem, bid))
        return bins, loads

    def first_fit_decreasing(order):
        bins = []
        loads = []
        rems = []
        for size, idx in order:
            placed = False
            for b in range(len(bins)):
                if rems[b] >= size:
                    bins[b].append(idx)
                    loads[b] += size
                    rems[b] -= size
                    placed = True
                    break
            if not placed:
                bins.append([idx])
                loads.append(size)
                rems.append(capacity - size)
        return bins, loads

    descending = sorted(indexed, key=lambda x: (-x[0], x[1]))

    best_bins, best_loads = best_fit_decreasing(descending)

    if time.time() < deadline and len(best_bins) > lower_bound:
        ffd_bins, ffd_loads = first_fit_decreasing(descending)
        if len(ffd_bins) < len(best_bins):
            best_bins, best_loads = ffd_bins, ffd_loads

    # Try a few deterministic tie-breaking variants, useful when many equal/near-equal sizes exist.
    if time.time() < deadline and len(best_bins) > lower_bound:
        asc_tie = sorted(indexed, key=lambda x: (-x[0], -x[1]))
        b, l = best_fit_decreasing(asc_tie)
        if len(b) < len(best_bins):
            best_bins, best_loads = b, l

    if time.time() < deadline and len(best_bins) > lower_bound:
        # Slight randomized perturbation among same/close sizes, bounded tightly.
        rnd = random.Random(1234567 + n + capacity)
        attempts = 3 if n <= 2000 else 1
        for _ in range(attempts):
            if time.time() >= deadline or len(best_bins) <= lower_bound:
                break
            arr = descending[:]
            # Shuffle blocks of equal sizes only: keeps descending quality, changes ties.
            i = 0
            while i < n:
                j = i + 1
                while j < n and arr[j][0] == arr[i][0]:
                    j += 1
                if j - i > 1:
                    block = arr[i:j]
                    rnd.shuffle(block)
                    arr[i:j] = block
                i = j
            b, l = best_fit_decreasing(arr)
            if len(b) < len(best_bins):
                best_bins, best_loads = b, l

    bins = [x[:] for x in best_bins]
    loads = best_loads[:]

    size_by_idx = [0] + items[:]

    def try_eliminate_bin(target):
        if time.time() >= deadline:
            return False

        target_items = bins[target][:]
        if not target_items:
            return True

        # Larger items first greatly prunes the search.
        target_items.sort(key=lambda idx: -size_by_idx[idx])

        other_bins = []
        other_loads = []
        old_to_new = {}
        for i, b in enumerate(bins):
            if i != target:
                old_to_new[i] = len(other_bins)
                other_bins.append(b)
                other_loads.append(loads[i])

        residual = [capacity - x for x in other_loads]

        # Quick necessary checks.
        for idx in target_items:
            if size_by_idx[idx] > max(residual) if residual else True:
                return False
        if sum(size_by_idx[i] for i in target_items) > sum(residual):
            return False

        assignment = [-1] * len(target_items)
        nodes = [0]
        max_nodes = 4000 if len(target_items) <= 18 else 1200

        def dfs(pos):
            nodes[0] += 1
            if nodes[0] > max_nodes or time.time() >= deadline:
                return False
            if pos == len(target_items):
                return True

            idx = target_items[pos]
            s = size_by_idx[idx]

            candidates = []
            for bi, r in enumerate(residual):
                if r >= s:
                    candidates.append((r - s, bi))
            if not candidates:
                return False

            # Best-fit order and skip equivalent residual capacities.
            candidates.sort()
            seen_residuals = set()

            for _, bi in candidates:
                r = residual[bi]
                if r in seen_residuals:
                    continue
                seen_residuals.add(r)

                residual[bi] -= s
                assignment[pos] = bi
                if dfs(pos + 1):
                    return True
                residual[bi] += s
                assignment[pos] = -1

            return False

        if not dfs(0):
            return False

        for pos, bi in enumerate(assignment):
            idx = target_items[pos]
            other_bins[bi].append(idx)
            other_loads[bi] += size_by_idx[idx]

        bins[:] = other_bins
        loads[:] = other_loads
        return True

    # Local improvement: repeatedly try to remove sparse bins.
    while time.time() < deadline and len(bins) > lower_bound:
        improved = False
        order = sorted(range(len(bins)), key=lambda i: (loads[i], len(bins[i])))
        for t in order:
            if time.time() >= deadline or len(bins) <= lower_bound:
                break
            if t >= len(bins):
                continue
            if try_eliminate_bin(t):
                improved = True
                break
        if not improved:
            break

    # Final safety validation/reconstruction if needed.
    seen = set()
    final_bins = []
    valid = True
    for b in bins:
        load = 0
        nb = []
        for idx in b:
            if idx < 1 or idx > n or idx in seen:
                valid = False
                break
            seen.add(idx)
            load += size_by_idx[idx]
            nb.append(idx)
        if load > capacity:
            valid = False
        if nb:
            final_bins.append(nb)
        if not valid:
            break

    if not valid or len(seen) != n:
        # Guaranteed valid fallback for normal instances.
        final_bins, _ = best_fit_decreasing(descending)

    return {
        "num_bins": len(final_bins),
        "bins": final_bins
    }