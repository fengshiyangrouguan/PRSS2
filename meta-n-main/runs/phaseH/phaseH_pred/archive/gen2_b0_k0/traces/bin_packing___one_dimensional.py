def solve(**kwargs):
    import time
    import random
    import math

    start_time = time.time()
    deadline = start_time + 9.2

    capacity = int(kwargs.get("bin_capacity"))
    items = list(kwargs.get("items", []))
    n = int(kwargs.get("num_items", len(items)))

    if n == 0:
        return {"num_bins": 0, "bins": []}

    indexed_items = [(items[i], i + 1) for i in range(n)]

    def best_fit(order):
        bins = []
        loads = []
        for size, idx in order:
            best_j = -1
            best_rem = capacity + 1
            for j, load in enumerate(loads):
                rem = capacity - load
                if size <= rem and rem - size < best_rem:
                    best_rem = rem - size
                    best_j = j
            if best_j == -1:
                bins.append([idx])
                loads.append(size)
            else:
                bins[best_j].append(idx)
                loads[best_j] += size
        return bins, loads

    def first_fit(order):
        bins = []
        loads = []
        for size, idx in order:
            placed = False
            for j, load in enumerate(loads):
                if load + size <= capacity:
                    bins[j].append(idx)
                    loads[j] += size
                    placed = True
                    break
            if not placed:
                bins.append([idx])
                loads.append(size)
        return bins, loads

    # Initial deterministic solutions.
    dec_order = sorted(indexed_items, key=lambda x: (-x[0], x[1]))
    inc_order = sorted(indexed_items, key=lambda x: (x[0], x[1]))

    best_bins, best_loads = best_fit(dec_order)
    for cand_bins, cand_loads in (
        first_fit(dec_order),
        best_fit(inc_order),
        first_fit(inc_order),
    ):
        if len(cand_bins) < len(best_bins):
            best_bins, best_loads = cand_bins, cand_loads

    total_size = sum(items)
    lower_bound = max((total_size + capacity - 1) // capacity, max(items) // capacity if items else 0)

    # Exact branch-and-bound for smaller instances.
    def exact_with_k(k):
        sorted_items = sorted([(items[i], i + 1) for i in range(n)], reverse=True)
        rem = [capacity] * k
        assign = [[] for _ in range(k)]
        calls = [0]

        # Simple suffix sum pruning.
        suffix = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix[i] = suffix[i + 1] + sorted_items[i][0]

        def dfs(pos):
            calls[0] += 1
            if calls[0] & 2047 == 0:
                if time.time() > deadline:
                    raise TimeoutError
            if pos == n:
                return True
            if sum(rem) < suffix[pos]:
                return False

            size, idx = sorted_items[pos]
            seen_remainders = set()

            # Try tighter bins first.
            order_bins = sorted(range(k), key=lambda b: rem[b])
            for b in order_bins:
                if rem[b] < size:
                    continue
                if rem[b] in seen_remainders:
                    continue
                seen_remainders.add(rem[b])

                rem[b] -= size
                assign[b].append(idx)

                if dfs(pos + 1):
                    return True

                assign[b].pop()
                rem[b] += size

                # Symmetry: if item did not fit into an empty bin successfully,
                # no need to try other empty bins.
                if rem[b] == capacity:
                    break
            return False

        try:
            if dfs(0):
                return [bin_[:] for bin_ in assign if bin_]
        except TimeoutError:
            return None
        return None

    if n <= 55 and lower_bound < len(best_bins):
        k = lower_bound
        while k < len(best_bins) and time.time() < deadline:
            exact = exact_with_k(k)
            if exact is not None:
                best_bins = exact
                best_loads = [sum(items[i - 1] for i in b) for b in best_bins]
                break
            k += 1

    # Try to eliminate bins by redistributing their contents.
    def try_remove_one_bin(bins, loads):
        m = len(bins)
        if m <= lower_bound:
            return bins, loads, False

        candidate_order = sorted(range(m), key=lambda j: (loads[j], len(bins[j])))

        for rem_bin_idx in candidate_order:
            if time.time() > deadline:
                return bins, loads, False

            moving = [(items[idx - 1], idx) for idx in bins[rem_bin_idx]]
            moving.sort(reverse=True)

            new_bins = [bins[j][:] for j in range(m) if j != rem_bin_idx]
            new_loads = [loads[j] for j in range(m) if j != rem_bin_idx]
            spaces = [capacity - x for x in new_loads]

            # Quick necessary check.
            if sum(size for size, _ in moving) > sum(spaces):
                continue
            if moving and moving[0][0] > max(spaces, default=0):
                continue

            placement = []

            def dfs_move(pos):
                if pos == len(moving):
                    return True
                if (pos & 7) == 0 and time.time() > deadline:
                    raise TimeoutError

                size, idx = moving[pos]
                seen = set()
                order = sorted(range(len(spaces)), key=lambda b: spaces[b])
                for b in order:
                    if spaces[b] >= size and spaces[b] not in seen:
                        seen.add(spaces[b])
                        spaces[b] -= size
                        new_loads[b] += size
                        new_bins[b].append(idx)
                        placement.append((b, idx))
                        if dfs_move(pos + 1):
                            return True
                        placement.pop()
                        new_bins[b].pop()
                        new_loads[b] -= size
                        spaces[b] += size
                return False

            try:
                if dfs_move(0):
                    return new_bins, new_loads, True
            except TimeoutError:
                return bins, loads, False

        return bins, loads, False

    improved = True
    while improved and len(best_bins) > lower_bound and time.time() < deadline:
        best_bins, best_loads, improved = try_remove_one_bin(best_bins, best_loads)

    # Randomized multi-start with subsequent bin elimination.
    rng = random.Random(1234567 + n + capacity)
    while time.time() < deadline and len(best_bins) > lower_bound:
        if n <= 1:
            break

        # Randomized decreasing order: mostly by size, with mild perturbation.
        if rng.random() < 0.5:
            order = dec_order[:]
            block = max(2, int(math.sqrt(n)))
            for s in range(0, n, block):
                sub = order[s:s + block]
                rng.shuffle(sub)
                order[s:s + block] = sub
        else:
            order = indexed_items[:]
            order.sort(key=lambda x: (-(x[0] * (0.85 + 0.3 * rng.random())), rng.random()))

        cand_bins, cand_loads = best_fit(order)
        if len(cand_bins) <= len(best_bins):
            improved2 = True
            while improved2 and len(cand_bins) > lower_bound and time.time() < deadline:
                cand_bins, cand_loads, improved2 = try_remove_one_bin(cand_bins, cand_loads)
            if len(cand_bins) < len(best_bins):
                best_bins, best_loads = cand_bins, cand_loads

    # Final safety validation/repair fallback.
    seen = []
    valid = True
    for b in best_bins:
        load = 0
        for idx in b:
            if idx < 1 or idx > n:
                valid = False
            else:
                load += items[idx - 1]
                seen.append(idx)
        if load > capacity:
            valid = False
    if len(seen) != n or len(set(seen)) != n:
        valid = False

    if not valid:
        best_bins, best_loads = best_fit(dec_order)

    return {
        "num_bins": len(best_bins),
        "bins": best_bins
    }