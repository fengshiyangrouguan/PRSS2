def solve(**kwargs):
    """
    Heuristic solver for the restricted single-machine common due date scheduling
    problem. Returns a valid 1-based permutation of the jobs.
    """
    import time
    import random

    jobs = kwargs.get('jobs', [])
    h = kwargs.get('h', 0.6)
    n = len(jobs)

    if n <= 1:
        return {'schedule': list(range(1, n + 1))}

    start_time = time.time()
    deadline = start_time + 9.15

    ps = [j[0] for j in jobs]
    aa = [j[1] for j in jobs]
    bb = [j[2] for j in jobs]

    total_p = sum(ps)
    d = int(total_p * h)

    def penalty(order):
        t = 0
        val = 0
        for idx in order:
            t += ps[idx]
            if t < d:
                val += aa[idx] * (d - t)
            elif t > d:
                val += bb[idx] * (t - d)
        return val

    all_idx = list(range(n))

    def make_v_shape(early_jobs):
        early_set = set(early_jobs)
        early = list(early_set)
        tardy = [i for i in all_idx if i not in early_set]

        early.sort(
            key=lambda i: (ps[i] / (aa[i] if aa[i] else 1e-9), ps[i]),
            reverse=True
        )
        tardy.sort(
            key=lambda i: (ps[i] / (bb[i] if bb[i] else 1e-9), -bb[i])
        )
        return early + tardy

    candidates = []

    candidates.append(all_idx[:])
    candidates.append(sorted(all_idx, key=lambda i: ps[i]))
    candidates.append(sorted(all_idx, key=lambda i: ps[i], reverse=True))
    candidates.append(sorted(all_idx, key=lambda i: ps[i] / (bb[i] if bb[i] else 1e-9)))
    candidates.append(sorted(all_idx, key=lambda i: ps[i] / (aa[i] if aa[i] else 1e-9), reverse=True))
    candidates.append(sorted(all_idx, key=lambda i: ps[i] / (aa[i] + bb[i] + 1e-9)))

    selection_orders = [
        sorted(all_idx, key=lambda i: (bb[i] / (aa[i] + 1e-9), -ps[i]), reverse=True),
        sorted(all_idx, key=lambda i: ((bb[i] + 1) / (aa[i] + 1), -ps[i]), reverse=True),
        sorted(all_idx, key=lambda i: (bb[i] - aa[i], -ps[i]), reverse=True),
        sorted(all_idx, key=lambda i: (bb[i] * ps[i], -aa[i]), reverse=True),
        sorted(all_idx, key=lambda i: (aa[i] / (bb[i] + 1e-9), ps[i])),
        sorted(all_idx, key=lambda i: ps[i], reverse=True),
    ]

    for order in selection_orders:
        load = 0
        early = []
        for i in order:
            if load + ps[i] <= d:
                early.append(i)
                load += ps[i]
        candidates.append(make_v_shape(early))

        load = 0
        early = []
        for i in order:
            if load + ps[i] <= d or load < d * 0.92:
                early.append(i)
                load += ps[i]
            if load >= d:
                break
        candidates.append(make_v_shape(early))

    if n <= 170 and time.time() < deadline:
        add_order = sorted(all_idx, key=lambda i: ps[i] * (aa[i] + bb[i]), reverse=True)
        seq = []
        complete = True

        for job in add_order:
            if time.time() > deadline:
                complete = False
                break

            best_seq = None
            best_val = None

            for pos in range(len(seq) + 1):
                trial = seq[:pos] + [job] + seq[pos:]
                val = penalty(trial)

                if best_val is None or val < best_val:
                    best_val = val
                    best_seq = trial

            seq = best_seq

        if complete and len(seq) == n:
            candidates.append(seq)

    best = None
    best_val = None
    seen = set()

    for cand in candidates:
        if len(cand) != n:
            continue

        key = tuple(cand)
        if key in seen:
            continue

        seen.add(key)
        val = penalty(cand)

        if best_val is None or val < best_val:
            best_val = val
            best = cand[:]

    if best is None:
        best = all_idx[:]
        best_val = penalty(best)

    def adjacent_improve(seq, val):
        passes = 0
        improved = True

        while improved and passes < 25 and time.time() < deadline:
            improved = False
            passes += 1

            for i in range(n - 1):
                if time.time() > deadline:
                    break

                seq[i], seq[i + 1] = seq[i + 1], seq[i]
                new_val = penalty(seq)

                if new_val < val:
                    val = new_val
                    improved = True
                else:
                    seq[i], seq[i + 1] = seq[i + 1], seq[i]

        return seq, val

    best, best_val = adjacent_improve(best, best_val)

    loops = 0
    while time.time() < deadline and loops < 4:
        loops += 1
        improved = False

        t = 0
        cross = 0
        for k, idx in enumerate(best):
            t += ps[idx]
            if t >= d:
                cross = k
                break

        for i in range(n):
            if time.time() > deadline:
                break

            job = best[i]
            base = best[:i] + best[i + 1:]

            if n <= 90:
                positions = range(n)
            else:
                positions_set = {
                    0, n - 1,
                    i, max(0, i - 1), min(n - 1, i + 1),
                    cross, max(0, cross - 1), min(n - 1, cross + 1)
                }

                for off in (2, 3, 5, 8, 13, 21):
                    positions_set.add(max(0, i - off))
                    positions_set.add(min(n - 1, i + off))
                    positions_set.add(max(0, cross - off))
                    positions_set.add(min(n - 1, cross + off))

                positions = sorted(positions_set)

            local_best_pos = i
            local_best_val = best_val

            for pos in positions:
                if pos == i:
                    continue

                trial = base[:pos] + [job] + base[pos:]
                val = penalty(trial)

                if val < local_best_val:
                    local_best_val = val
                    local_best_pos = pos

            if local_best_val < best_val:
                best = base[:local_best_pos] + [job] + base[local_best_pos:]
                best_val = local_best_val
                improved = True

        if not improved:
            break

        best, best_val = adjacent_improve(best, best_val)

    rnd = random.Random(1234567)
    attempts = 0

    while time.time() < deadline and attempts < 20 and n <= 250:
        attempts += 1
        seq = best[:]

        for _ in range(max(1, n // 25)):
            i = rnd.randrange(n)
            j = rnd.randrange(n)
            seq[i], seq[j] = seq[j], seq[i]

        val = penalty(seq)
        seq, val = adjacent_improve(seq, val)

        if val < best_val:
            best = seq
            best_val = val

    return {'schedule': [i + 1 for i in best]}