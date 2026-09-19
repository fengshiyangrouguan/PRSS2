def solve(**kwargs):
    """
    Solves the restricted single-machine common due date scheduling problem.
    Returns {'schedule': [...]} with 1-based job indices.
    """
    import time
    import random

    jobs = kwargs.get("jobs", [])
    h = kwargs.get("h", 0.6)
    n = len(jobs)

    if n == 0:
        return {"schedule": []}
    if n == 1:
        return {"schedule": [1]}

    start_time = time.time()
    deadline = start_time + 9.0

    p = [int(j[0]) for j in jobs]
    a = [int(j[1]) for j in jobs]
    b = [int(j[2]) for j in jobs]

    total_p = sum(p)
    d = int(total_p * h)

    indices = list(range(n))

    def penalty_of_schedule(schedule):
        t = 0
        cost = 0
        for idx in schedule:
            t += p[idx]
            if t < d:
                cost += a[idx] * (d - t)
            elif t > d:
                cost += b[idx] * (t - d)
        return cost

    def make_schedule(early_set):
        early = [i for i in indices if early_set[i]]
        late = [i for i in indices if not early_set[i]]

        # V-shaped property heuristic:
        # Early side: decreasing p/a.
        # Tardy side: increasing p/b.
        early.sort(key=lambda i: (-(p[i] / (a[i] if a[i] > 0 else 1e-9)), -p[i], i))
        late.sort(key=lambda i: ((p[i] / (b[i] if b[i] > 0 else 1e-9)), p[i], i))
        return early + late

    def evaluate_set(early_set):
        return penalty_of_schedule(make_schedule(early_set))

    def build_by_order(order):
        early_set = [False] * n
        s = 0
        for i in order:
            if s + p[i] <= d:
                early_set[i] = True
                s += p[i]
        return early_set, s

    candidates = []

    orders = []

    orders.append(sorted(indices, key=lambda i: (-(p[i] / (a[i] if a[i] > 0 else 1e-9)), -a[i], i)))
    orders.append(sorted(indices, key=lambda i: (-(a[i] / (p[i] if p[i] > 0 else 1e-9)), -a[i], i)))
    orders.append(sorted(indices, key=lambda i: (-a[i], p[i], i)))
    orders.append(sorted(indices, key=lambda i: (b[i], -a[i], p[i], i)))
    orders.append(sorted(indices, key=lambda i: (-(a[i] / ((a[i] + b[i]) if (a[i] + b[i]) > 0 else 1e-9)), p[i], i)))
    orders.append(sorted(indices, key=lambda i: (-(a[i] - b[i]), p[i], i)))
    orders.append(sorted(indices, key=lambda i: (p[i], -a[i], i)))

    for order in orders:
        es, sp = build_by_order(order)
        candidates.append((es, sp))

    # Greedy randomized-ish deterministic variants: start with high earliness benefit
    if n <= 2000:
        rng = random.Random(12345)
        base = sorted(indices, key=lambda i: (-(a[i] / (p[i] if p[i] else 1)), b[i], i))
        for _ in range(4):
            arr = base[:]
            # small deterministic shuffle among nearby elements
            for k in range(n):
                j = k + rng.randrange(min(8, n - k))
                arr[k], arr[j] = arr[j], arr[k]
            es, sp = build_by_order(arr)
            candidates.append((es, sp))

    best_set = None
    best_sum = 0
    best_cost = None

    for es, sp in candidates:
        c = evaluate_set(es)
        if best_cost is None or c < best_cost:
            best_cost = c
            best_set = es[:]
            best_sum = sp

    # Local improvement on early/tardy partition.
    # Keep early processing sum <= d, which is usually desirable for restricted CDD.
    def local_improve(early_set, early_sum, current_cost):
        improved = True
        passes = 0

        while improved and passes < 6 and time.time() < deadline:
            improved = False
            passes += 1

            # Single-job flips
            order = sorted(indices, key=lambda i: -max(a[i], b[i]))
            for i in order:
                if time.time() >= deadline:
                    break

                if early_set[i]:
                    early_set[i] = False
                    nc = evaluate_set(early_set)
                    if nc < current_cost:
                        current_cost = nc
                        early_sum -= p[i]
                        improved = True
                    else:
                        early_set[i] = True
                else:
                    if early_sum + p[i] <= d:
                        early_set[i] = True
                        nc = evaluate_set(early_set)
                        if nc < current_cost:
                            current_cost = nc
                            early_sum += p[i]
                            improved = True
                        else:
                            early_set[i] = False

            if time.time() >= deadline:
                break

            # Pair exchange: one early out, one tardy in
            if n <= 600:
                early_jobs = [i for i in indices if early_set[i]]
                late_jobs = [i for i in indices if not early_set[i]]

                early_jobs.sort(key=lambda i: (a[i] - b[i], -p[i]))
                late_jobs.sort(key=lambda i: (-(a[i] - b[i]), p[i]))

                max_e = min(len(early_jobs), 80)
                max_l = min(len(late_jobs), 80)

                for x in early_jobs[:max_e]:
                    if time.time() >= deadline:
                        break
                    for y in late_jobs[:max_l]:
                        if early_sum - p[x] + p[y] > d:
                            continue
                        early_set[x] = False
                        early_set[y] = True
                        nc = evaluate_set(early_set)
                        if nc < current_cost:
                            current_cost = nc
                            early_sum = early_sum - p[x] + p[y]
                            improved = True
                            break
                        early_set[x] = True
                        early_set[y] = False
                    if improved:
                        break

        return early_set, early_sum, current_cost

    if time.time() < deadline:
        best_set, best_sum, best_cost = local_improve(best_set, best_sum, best_cost)

    best_schedule = make_schedule(best_set)
    best_cost = penalty_of_schedule(best_schedule)

    # Final permutation-level adjacent-swap cleanup.
    # This can improve cases where the strict V-order heuristic is slightly off.
    if n <= 800 and time.time() < deadline:
        sched = best_schedule[:]
        improved = True
        passes = 0
        while improved and passes < 4 and time.time() < deadline:
            improved = False
            passes += 1
            for k in range(n - 1):
                if time.time() >= deadline:
                    break
                sched[k], sched[k + 1] = sched[k + 1], sched[k]
                nc = penalty_of_schedule(sched)
                if nc < best_cost:
                    best_cost = nc
                    improved = True
                else:
                    sched[k], sched[k + 1] = sched[k + 1], sched[k]
        best_schedule = sched

    return {"schedule": [i + 1 for i in best_schedule]}