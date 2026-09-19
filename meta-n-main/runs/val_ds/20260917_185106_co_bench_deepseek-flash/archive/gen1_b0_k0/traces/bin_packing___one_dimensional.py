def solve(**kwargs):
    import time
    import math
    import random

    start_time = time.monotonic()
    deadline = start_time + 9.0

    capacity = kwargs.get('bin_capacity')
    items = kwargs.get('items', [])
    n = len(items)

    if n == 0:
        return {'num_bins': 0, 'bins': []}

    sizes = list(items)

    total = sum(sizes)
    LB = max(1, (total + capacity - 1) // capacity)
    large_count = sum(1 for s in sizes if 2 * s > capacity)
    LB = max(LB, large_count)

    def bfd(pairs):
        bins = []
        for s, idx in pairs:
            best_i = -1
            best_rem = capacity + 1
            for i, (rem, lst) in enumerate(bins):
                if rem >= s and rem < best_rem:
                    best_rem = rem
                    best_i = i
            if best_i == -1:
                bins.append([capacity - s, [idx]])
            else:
                bins[best_i][0] -= s
                bins[best_i][1].append(idx)
        return bins

    def ffd(pairs):
        bins = []
        for s, idx in pairs:
            placed = False
            for i, (rem, lst) in enumerate(bins):
                if rem >= s:
                    bins[i][0] -= s
                    bins[i][1].append(idx)
                    placed = True
                    break
            if not placed:
                bins.append([capacity - s, [idx]])
        return bins

    sorted_pairs = sorted([(sizes[i], i) for i in range(n)], key=lambda x: (-x[0], x[1]))

    bins_bfd = bfd(sorted_pairs)
    best_bins = [lst for _, lst in bins_bfd]
    best_loads = [sum(sizes[i] for i in lst) for lst in best_bins]
    best_count = len(best_bins)

    if best_count == LB:
        return {'num_bins': best_count, 'bins': [[i+1 for i in bin] for bin in best_bins]}

    bins_ffd = ffd(sorted_pairs)
    if len(bins_ffd) < best_count:
        best_bins = [lst for _, lst in bins_ffd]
        best_loads = [sum(sizes[i] for i in lst) for lst in best_bins]
        best_count = len(best_bins)
        if best_count == LB:
            return {'num_bins': best_count, 'bins': [[i+1 for i in bin] for bin in best_bins]}

    rng = random.Random(12345)
    trials = 50 if n <= 200 else 10
    for _ in range(trials):
        if time.monotonic() > deadline:
            break
        keyed = [(sizes[i], i, rng.random()) for i in range(n)]
        keyed.sort(key=lambda x: (-x[0], x[2]))
        pairs = [(s, i) for s, i, _ in keyed]
        bins = bfd(pairs)
        if len(bins) < best_count:
            best_bins = [lst for _, lst in bins]
            best_loads = [sum(sizes[i] for i in lst) for lst in best_bins]
            best_count = len(best_bins)
            if best_count == LB:
                break

    if best_count == LB:
        return {'num_bins': best_count, 'bins': [[i+1 for i in bin] for bin in best_bins]}

    def sa_fixed(bins, loads, capacity, sizes, deadline):
        K = len(bins)
        n = len(sizes)
        if K == 0:
            return bins if n == 0 else None
        overflow = sum(max(0, l - capacity) for l in loads)
        if overflow == 0:
            return bins

        avg_item = sum(sizes) / n if n > 0 else 1.0
        T0 = max(1.0, avg_item * 0.5)
        T_min = 0.01
        start = time.monotonic()
        total_time = deadline - start
        if total_time <= 0:
            return None

        best_bins = [b[:] for b in bins]
        best_loads = loads[:]
        best_overflow = overflow

        cap = capacity
        items = sizes

        while True:
            now = time.monotonic()
            if now >= deadline:
                break
            elapsed = now - start
            if total_time > 0:
                frac = min(1.0, elapsed / total_time)
                T = T0 * (1 - frac) + T_min * frac
            else:
                T = T_min
            for _ in range(500):
                if overflow == 0:
                    return bins
                b1 = random.randrange(K)
                if loads[b1] <= cap:
                    for _ in range(5):
                        j = random.randrange(K)
                        if loads[j] > cap:
                            b1 = j
                            break
                if not bins[b1]:
                    continue
                item_idx = random.randrange(len(bins[b1]))
                item = bins[b1][item_idx]
                size = items[item]

                if random.random() < 0.7:
                    b2 = random.randrange(K)
                    if b2 == b1:
                        continue
                    old_over = max(0, loads[b1] - cap) + max(0, loads[b2] - cap)
                    new_load1 = loads[b1] - size
                    new_load2 = loads[b2] + size
                    new_over = max(0, new_load1 - cap) + max(0, new_load2 - cap)
                    delta = new_over - old_over
                    if delta <= 0 or random.random() < math.exp(-delta / max(T, 1e-9)):
                        bins[b1].pop(item_idx)
                        bins[b2].append(item)
                        loads[b1] = new_load1
                        loads[b2] = new_load2
                        overflow += delta
                else:
                    b2 = random.randrange(K)
                    if b2 == b1 or not bins[b2]:
                        continue
                    i2 = random.randrange(len(bins[b2]))
                    item2 = bins[b2][i2]
                    size2 = items[item2]
                    old_over = max(0, loads[b1] - cap) + max(0, loads[b2] - cap)
                    new_load1 = loads[b1] - size + size2
                    new_load2 = loads[b2] - size2 + size
                    new_over = max(0, new_load1 - cap) + max(0, new_load2 - cap)
                    delta = new_over - old_over
                    if delta <= 0 or random.random() < math.exp(-delta / max(T, 1e-9)):
                        bins[b1][item_idx] = item2
                        bins[b2][i2] = item
                        loads[b1] = new_load1
                        loads[b2] = new_load2
                        overflow += delta
                if overflow < best_overflow:
                    best_overflow = overflow
                    best_bins = [b[:] for b in bins]
                    best_loads = loads[:]
                    if best_overflow == 0:
                        return best_bins
        if best_overflow == 0:
            return best_bins
        return None

    while best_count > LB and time.monotonic() < deadline:
        improved = False
        targets = sorted(range(best_count), key=lambda i: best_loads[i])
        targets = targets[:10]
        remaining_time = deadline - time.monotonic()
        if remaining_time <= 0:
            break
        time_per_target = min(1.0, remaining_time / len(targets))
        for t in targets:
            if time.monotonic() >= deadline:
                break
            target_items = best_bins[t][:]
            new_bins = []
            new_loads = []
            for i in range(best_count):
                if i != t:
                    new_bins.append(best_bins[i][:])
                    new_loads.append(best_loads[i])
            target_items.sort(key=lambda idx: -sizes[idx])
            for idx in target_items:
                s = sizes[idx]
                best_i = -1
                best_rem = None
                for i, load in enumerate(new_loads):
                    if load + s <= capacity:
                        rem = capacity - (load + s)
                        if best_rem is None or rem < best_rem:
                            best_rem = rem
                            best_i = i
                if best_i == -1:
                    best_i = min(range(len(new_loads)), key=lambda i: new_loads[i])
                new_bins[best_i].append(idx)
                new_loads[best_i] += s

            overflow = sum(max(0, l - capacity) for l in new_loads)
            if overflow == 0:
                best_bins = new_bins
                best_loads = new_loads
                best_count = len(best_bins)
                improved = True
                break

            slice_end = min(deadline, time.monotonic() + time_per_target)
            reduced = sa_fixed(new_bins, new_loads, capacity, sizes, slice_end)
            if reduced is not None:
                best_bins = reduced
                best_loads = [sum(sizes[i] for i in lst) for lst in best_bins]
                best_count = len(best_bins)
                improved = True
                break
        if not improved:
            break

    return {
        'num_bins': best_count,
        'bins': [[i+1 for i in bin] for bin in best_bins]
    }