def solve(**kwargs):
    import time
    import random

    start_time = time.time()
    deadline = start_time + 8.5

    n = int(kwargs["num_planes"])
    m = int(kwargs["num_runways"])
    planes = kwargs["planes"]
    sep = kwargs["separation"]

    if n == 0:
        return {"schedule": {}}

    earliest = [float(p["earliest"]) for p in planes]
    target = [float(p["target"]) for p in planes]
    latest = [float(p["latest"]) for p in planes]
    pe = [float(p["penalty_early"]) for p in planes]
    pl = [float(p["penalty_late"]) for p in planes]
    appearance = [float(p.get("appearance", 0.0)) for p in planes]

    def penalty(i, t):
        if t < target[i]:
            return (target[i] - t) * pe[i]
        return (t - target[i]) * pl[i]

    def schedule_sequence(seq, improve=True):
        """Return (times, cost) for a fixed runway order, or None if infeasible."""
        k = len(seq)
        if k == 0:
            return [], 0.0

        times = [0.0] * k

        prev = seq[0]
        t = max(earliest[prev], appearance[prev])
        if t > latest[prev] + 1e-9:
            return None
        times[0] = t

        for pos in range(1, k):
            i = seq[pos]
            p = seq[pos - 1]
            t = max(earliest[i], appearance[i], times[pos - 1] + float(sep[p][i]))
            if t > latest[i] + 1e-9:
                return None
            times[pos] = t

        if improve:
            # Push early aircraft to the right where possible, without violating successors.
            for pos in range(k - 1, -1, -1):
                i = seq[pos]
                upper = latest[i]
                if pos + 1 < k:
                    j = seq[pos + 1]
                    upper = min(upper, times[pos + 1] - float(sep[i][j]))
                if times[pos] < target[i]:
                    nt = min(target[i], upper)
                    if nt > times[pos]:
                        times[pos] = nt

        cost = 0.0
        for pos, i in enumerate(seq):
            t = times[pos]
            if t < earliest[i] - 1e-7 or t > latest[i] + 1e-7:
                return None
            if pos > 0:
                p = seq[pos - 1]
                if t < times[pos - 1] + float(sep[p][i]) - 1e-7:
                    return None
            cost += penalty(i, t)

        return times, cost

    def total_solution_cost(runways):
        total = 0.0
        all_times = []
        for seq in runways:
            res = schedule_sequence(seq, True)
            if res is None:
                return None, None
            ts, c = res
            all_times.append(ts)
            total += c
        return total, all_times

    def greedy_build(order):
        runways = [[] for _ in range(m)]
        runway_costs = [0.0] * m
        runway_times = [[] for _ in range(m)]

        for plane in order:
            if time.time() > deadline:
                return None

            best = None
            best_r = None
            best_pos = None
            best_seq = None
            best_times = None
            best_cost = None

            for r in range(m):
                seq = runways[r]
                for pos in range(len(seq) + 1):
                    cand = seq[:pos] + [plane] + seq[pos:]
                    res = schedule_sequence(cand, True)
                    if res is None:
                        continue
                    ts, c = res
                    delta = c - runway_costs[r]

                    # Small tie-breaker favoring landing close to target and balanced runways.
                    landing_pos = cand.index(plane)
                    close = abs(ts[landing_pos] - target[plane])
                    score = delta + 1e-6 * close + 1e-7 * len(cand)

                    if best is None or score < best:
                        best = score
                        best_r = r
                        best_pos = pos
                        best_seq = cand
                        best_times = ts
                        best_cost = c

            if best is None:
                return None

            runways[best_r] = best_seq
            runway_times[best_r] = best_times
            runway_costs[best_r] = best_cost

        total, all_times = total_solution_cost(runways)
        if total is None:
            return None
        return runways, all_times, total

    indices = list(range(n))

    orderings = []

    orderings.append(sorted(indices, key=lambda i: (target[i], earliest[i], latest[i])))
    orderings.append(sorted(indices, key=lambda i: (latest[i], target[i], earliest[i])))
    orderings.append(sorted(indices, key=lambda i: (earliest[i], target[i], latest[i])))
    orderings.append(sorted(indices, key=lambda i: (appearance[i], target[i], latest[i])))
    orderings.append(sorted(indices, key=lambda i: (latest[i] - earliest[i], target[i])))
    orderings.append(sorted(indices, key=lambda i: (-(pe[i] + pl[i]), target[i])))
    orderings.append(sorted(indices, key=lambda i: (target[i] - earliest[i], latest[i])))

    best_runways = None
    best_times = None
    best_cost = None

    for order in orderings:
        if time.time() > deadline:
            break
        sol = greedy_build(order)
        if sol is not None:
            rw, ts, c = sol
            if best_cost is None or c < best_cost:
                best_runways, best_times, best_cost = rw, ts, c

    # Limited randomized restarts.
    rng = random.Random(1234567 + n * 97 + m)
    base = sorted(indices, key=lambda i: target[i])
    while time.time() < deadline:
        order = base[:]
        # Perturb target order locally.
        if n > 1:
            width = max(2, min(n, 7))
            for _ in range(max(1, n // 3)):
                a = rng.randrange(n)
                b = min(n - 1, max(0, a + rng.randrange(-width, width + 1)))
                order[a], order[b] = order[b], order[a]
        sol = greedy_build(order)
        if sol is not None:
            rw, ts, c = sol
            if best_cost is None or c < best_cost:
                best_runways, best_times, best_cost = rw, ts, c

    # Fallback: construct a simple earliest-time schedule if greedy insertion failed.
    if best_runways is None:
        runways = [[] for _ in range(m)]
        for idx, i in enumerate(sorted(indices, key=lambda x: (earliest[x], target[x]))):
            runways[idx % m].append(i)
        total, all_times = total_solution_cost(runways)
        if total is not None:
            best_runways, best_times = runways, all_times
        else:
            # Last-resort output: individually feasible times, may be infeasible only on very hard instances.
            schedule = {}
            for i in indices:
                t = min(max(target[i], earliest[i], appearance[i]), latest[i])
                schedule[i + 1] = {"landing_time": float(t), "runway": (i % m) + 1}
            return {"schedule": schedule}

    schedule = {}
    for r, seq in enumerate(best_runways):
        ts = best_times[r]
        for pos, i in enumerate(seq):
            schedule[i + 1] = {
                "landing_time": float(ts[pos]),
                "runway": int(r + 1),
            }

    # Ensure every plane is present.
    for i in indices:
        if i + 1 not in schedule:
            t = min(max(target[i], earliest[i], appearance[i]), latest[i])
            schedule[i + 1] = {"landing_time": float(t), "runway": 1}

    return {"schedule": schedule}