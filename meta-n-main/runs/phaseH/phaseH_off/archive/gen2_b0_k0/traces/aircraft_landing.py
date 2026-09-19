def solve(**kwargs):
    import time
    import random

    start_time = time.time()
    deadline = start_time + 8.7

    n = int(kwargs.get("num_planes", 0))
    m = max(1, int(kwargs.get("num_runways", 1)))
    planes = kwargs["planes"]
    sep = kwargs["separation"]

    E = [float(p["earliest"]) for p in planes]
    T = [float(p["target"]) for p in planes]
    L = [float(p["latest"]) for p in planes]
    PE = [float(p.get("penalty_early", 1.0)) for p in planes]
    PL = [float(p.get("penalty_late", 1.0)) for p in planes]

    def schedule_order(order):
        """Return ({plane_id0: time}, cost) for a fixed runway order, or (None, inf)."""
        k = len(order)
        if k == 0:
            return {}, 0.0

        earliest_feasible = [0.0] * k
        for pos, pid in enumerate(order):
            if pos == 0:
                val = E[pid]
            else:
                prev = order[pos - 1]
                val = max(E[pid], earliest_feasible[pos - 1] + float(sep[prev][pid]))
            if val > L[pid] + 1e-7:
                return None, float("inf")
            earliest_feasible[pos] = val

        latest_feasible = [0.0] * k
        for pos in range(k - 1, -1, -1):
            pid = order[pos]
            if pos == k - 1:
                val = L[pid]
            else:
                nxt = order[pos + 1]
                val = min(L[pid], latest_feasible[pos + 1] - float(sep[pid][nxt]))
            if val + 1e-7 < earliest_feasible[pos]:
                return None, float("inf")
            latest_feasible[pos] = val

        times_list = [0.0] * k
        cost = 0.0
        next_time = None
        next_pid = None

        for pos in range(k - 1, -1, -1):
            pid = order[pos]
            upper = latest_feasible[pos]
            if next_time is not None:
                upper = min(upper, next_time - float(sep[pid][next_pid]))

            if upper + 1e-7 < earliest_feasible[pos]:
                return None, float("inf")

            t = T[pid]
            if t < earliest_feasible[pos]:
                t = earliest_feasible[pos]
            elif t > upper:
                t = upper

            times_list[pos] = t

            if t < T[pid]:
                cost += (T[pid] - t) * PE[pid]
            else:
                cost += (t - T[pid]) * PL[pid]

            next_time = t
            next_pid = pid

        return {order[i]: times_list[i] for i in range(k)}, cost

    def eval_solution(orders):
        all_times = {}
        total_cost = 0.0
        for order in orders:
            td, c = schedule_order(order)
            if td is None:
                return None, float("inf")
            all_times.update(td)
            total_cost += c
        if len(all_times) != n:
            return None, float("inf")
        return all_times, total_cost

    def build_with_order(plane_order):
        orders = [[] for _ in range(m)]
        runway_costs = [0.0] * m

        for pid in plane_order:
            if time.time() > deadline:
                return None, float("inf")

            best = None
            best_key = None

            for r in range(m):
                base = orders[r]
                for pos in range(len(base) + 1):
                    cand = base[:pos] + [pid] + base[pos:]
                    _, c = schedule_order(cand)
                    if c == float("inf"):
                        continue

                    delta = c - runway_costs[r]
                    key = (delta, c, len(base), r, pos)
                    if best_key is None or key < best_key:
                        best_key = key
                        best = (r, cand, c)

            if best is None:
                return None, float("inf")

            r, cand, c = best
            orders[r] = cand
            runway_costs[r] = c

        times, cost = eval_solution(orders)
        if times is None:
            return None, float("inf")
        return (orders, times), cost

    indices = list(range(n))

    candidate_orders = [
        sorted(indices, key=lambda i: (T[i], E[i], L[i])),
        sorted(indices, key=lambda i: (L[i], T[i], E[i])),
        sorted(indices, key=lambda i: (E[i], T[i], L[i])),
        sorted(indices, key=lambda i: (L[i] - E[i], T[i])),
        sorted(indices, key=lambda i: (T[i] - E[i], L[i])),
        sorted(indices, key=lambda i: (-(PE[i] + PL[i]), T[i])),
    ]

    best_sol = None
    best_cost = float("inf")

    for order in candidate_orders:
        if time.time() > deadline:
            break
        sol, cost = build_with_order(order)
        if sol is not None and cost < best_cost:
            best_sol = sol
            best_cost = cost

    rng = random.Random(12345)
    base = sorted(indices, key=lambda i: T[i])

    while time.time() < deadline and n <= 120:
        noisy = base[:]
        noisy.sort(
            key=lambda i: (
                T[i] + rng.uniform(-1.0, 1.0) * max(1.0, 0.15 * (L[i] - E[i])),
                L[i],
            )
        )
        sol, cost = build_with_order(noisy)
        if sol is not None and cost < best_cost:
            best_sol = sol
            best_cost = cost

    if best_sol is None:
        orders = [[] for _ in range(m)]
        costs = [0.0] * m

        for pid in sorted(indices, key=lambda i: (E[i], L[i], T[i])):
            best = None
            for r in range(m):
                cand = orders[r] + [pid]
                _, c = schedule_order(cand)
                if c < float("inf"):
                    key = (c - costs[r], len(orders[r]), r)
                    if best is None or key < best[0]:
                        best = (key, r, cand, c)

            if best is not None:
                _, r, cand, c = best
                orders[r] = cand
                costs[r] = c
            else:
                r = min(range(m), key=lambda x: len(orders[x]))
                orders[r].append(pid)

        times, _ = eval_solution(orders)
        if times is not None:
            best_sol = (orders, times)

    schedule = {}

    if best_sol is not None:
        orders, times = best_sol
        runway_of = {}
        for r, order in enumerate(orders, start=1):
            for pid in order:
                runway_of[pid] = r

        for pid in range(n):
            schedule[pid + 1] = {
                "landing_time": float(times.get(pid, max(E[pid], min(T[pid], L[pid])))),
                "runway": int(runway_of.get(pid, 1)),
            }
    else:
        for pid in range(n):
            schedule[pid + 1] = {
                "landing_time": float(max(E[pid], min(T[pid], L[pid]))),
                "runway": int((pid % m) + 1),
            }

    return {"schedule": schedule}