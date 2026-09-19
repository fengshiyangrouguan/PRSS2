def solve(**kwargs):
    import time
    import math
    import random

    start_time = time.time()
    deadline = start_time + 8.7

    jobs = kwargs.get("jobs", [])
    h = kwargs.get("h", 0.6)
    n = len(jobs)

    if n == 0:
        return {"schedule": []}

    p = [int(x[0]) for x in jobs]
    a = [int(x[1]) for x in jobs]
    b = [int(x[2]) for x in jobs]
    total_p = sum(p)
    d = int(math.floor(total_p * h))

    def penalty_job(j, c):
        if c < d:
            return a[j] * (d - c)
        if c > d:
            return b[j] * (c - d)
        return 0

    def eval_order(order):
        t = 0
        val = 0
        for j in order:
            t += p[j]
            if t < d:
                val += a[j] * (d - t)
            elif t > d:
                val += b[j] * (t - d)
        return val

    # Exact dynamic programming for small instances.
    if n <= 17:
        m = 1 << n
        inf = 10**30
        dp = [inf] * m
        parent = [-1] * m
        prevmask = [-1] * m
        sump = [0] * m
        dp[0] = 0

        for mask in range(1, m):
            lb = mask & -mask
            j = lb.bit_length() - 1
            sump[mask] = sump[mask ^ lb] + p[j]

        for mask in range(m):
            base = dp[mask]
            if base >= inf:
                continue
            t0 = sump[mask]
            rem = ((m - 1) ^ mask)
            while rem:
                lb = rem & -rem
                j = lb.bit_length() - 1
                nm = mask | lb
                c = t0 + p[j]
                val = base + penalty_job(j, c)
                if val < dp[nm]:
                    dp[nm] = val
                    parent[nm] = j
                    prevmask[nm] = mask
                rem ^= lb

            if mask % 4096 == 0 and time.time() > deadline:
                break

        full = m - 1
        if parent[full] != -1:
            order = []
            mask = full
            while mask:
                j = parent[mask]
                order.append(j)
                mask = prevmask[mask]
            order.reverse()
            return {"schedule": [x + 1 for x in order]}

    def make_partition_order(key_func, reverse=True, fill_factor=1.0, allow_cross=False):
        idx = list(range(n))
        idx.sort(key=key_func, reverse=reverse)

        cap = int(d * fill_factor)
        early = []
        tardy = []
        used = 0

        for j in idx:
            if used + p[j] <= cap or (allow_cross and not early and used < cap):
                early.append(j)
                used += p[j]
            else:
                tardy.append(j)

        # Early part: reverse Smith rule for maximizing weighted completion.
        early.sort(key=lambda j: p[j] / (a[j] if a[j] > 0 else 1e-9), reverse=True)
        # Tardy part: Smith rule for minimizing weighted completion.
        tardy.sort(key=lambda j: p[j] / (b[j] if b[j] > 0 else 1e-9))
        return early + tardy

    candidates = []

    base = list(range(n))
    candidates.append(base[:])
    candidates.append(sorted(base, key=lambda j: p[j]))
    candidates.append(sorted(base, key=lambda j: p[j] / (b[j] if b[j] else 1e-9)))
    candidates.append(sorted(base, key=lambda j: p[j] / (a[j] if a[j] else 1e-9), reverse=True))
    candidates.append(sorted(base, key=lambda j: b[j] / (a[j] + b[j] + 1e-9), reverse=True))
    candidates.append(sorted(base, key=lambda j: (b[j] - a[j]), reverse=True))

    key_funcs = [
        lambda j: b[j] / (a[j] + 1e-9),
        lambda j: b[j] / (a[j] + b[j] + 1e-9),
        lambda j: b[j] * p[j],
        lambda j: b[j] - a[j],
        lambda j: b[j],
        lambda j: b[j] / (p[j] + 1e-9),
    ]

    for ff in (0.85, 0.95, 1.0, 1.05, 1.15):
        for kf in key_funcs:
            candidates.append(make_partition_order(kf, True, ff, False))
            if len(candidates) < 60:
                candidates.append(make_partition_order(kf, True, ff, True))

    best_order = None
    best_val = 10**40

    for cand in candidates:
        if time.time() > deadline:
            break
        val = eval_order(cand)
        if val < best_val:
            best_val = val
            best_order = cand[:]

    if best_order is None:
        best_order = list(range(n))
        best_val = eval_order(best_order)

    def adjacent_descent(order, val):
        improved = True
        passes = 0
        while improved and time.time() < deadline and passes < 50:
            improved = False
            passes += 1
            t = 0
            i = 0
            while i < n - 1:
                if time.time() > deadline:
                    break
                x = order[i]
                y = order[i + 1]
                c1 = t + p[x]
                c2 = c1 + p[y]
                old = penalty_job(x, c1) + penalty_job(y, c2)

                nc1 = t + p[y]
                nc2 = nc1 + p[x]
                new = penalty_job(y, nc1) + penalty_job(x, nc2)

                if new < old:
                    order[i], order[i + 1] = order[i + 1], order[i]
                    val += new - old
                    improved = True
                    # Prefix time after two jobs is unchanged.
                    t += p[x] + p[y]
                    i += 2
                else:
                    t += p[x]
                    i += 1
        return order, val

    # Improve promising deterministic candidates.
    for cand in candidates[:40]:
        if time.time() > deadline:
            break
        val = eval_order(cand)
        cand, val = adjacent_descent(cand[:], val)
        if val < best_val:
            best_val = val
            best_order = cand[:]

    # Randomized perturbation + adjacent descent for remaining time.
    rng = random.Random(1234567 + n + total_p)
    while time.time() < deadline:
        cur = best_order[:]

        # Light perturbation: random removals and insertions.
        moves = 1 + min(12, n // 20)
        for _ in range(moves):
            if n <= 1:
                break
            i = rng.randrange(n)
            job = cur.pop(i)
            j = rng.randrange(n)
            cur.insert(j, job)

        val = eval_order(cur)
        cur, val = adjacent_descent(cur, val)

        if val < best_val:
            best_val = val
            best_order = cur[:]

    return {"schedule": [j + 1 for j in best_order]}