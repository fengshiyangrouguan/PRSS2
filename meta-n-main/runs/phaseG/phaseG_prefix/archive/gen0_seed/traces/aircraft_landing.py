def solve(**kwargs):
    import time
    import math
    import random

    start_time = time.time()
    deadline = start_time + 8.5

    n = int(kwargs.get("num_planes", 0))
    m = max(1, int(kwargs.get("num_runways", 1)))
    planes = kwargs["planes"]
    sep = kwargs["separation"]

    E = [float(p["earliest"]) for p in planes]
    T = [float(p["target"]) for p in planes]
    L = [float(p["latest"]) for p in planes]
    PE = [float(p["penalty_early"]) for p in planes]
    PL = [float(p["penalty_late"]) for p in planes]

    def eval_sequence(seq):
        """Return (feasible, cost, times_for_seq) for a fixed runway order."""
        k = len(seq)
        if k == 0:
            return True, 0.0, []

        earliest = [0.0] * k
        for a, p in enumerate(seq):
            if a == 0:
                val = E[p]
            else:
                prev = seq[a - 1]
                val = max(E[p], earliest[a - 1] + float(sep[prev][p]))
            if val > L[p] + 1e-9:
                return False, 1e100, []
            earliest[a] = val

        latest = [0.0] * k
        for a in range(k - 1, -1, -1):
            p = seq[a]
            if a == k - 1:
                val = L[p]
            else:
                nxt = seq[a + 1]
                val = min(L[p], latest[a + 1] - float(sep[p][nxt]))
            if val + 1e-9 < earliest[a]:
                return False, 1e100, []
            latest[a] = val

        times = [0.0] * k
        cost = 0.0

        # Backward target-clamping gives a good feasible timing for this fixed order.
        upper = 1e100
        for a in range(k - 1, -1, -1):
            p = seq[a]
            hi = min(latest[a], upper)
            lo = earliest[a]
            if T[p] < lo:
                t = lo
            elif T[p] > hi:
                t = hi
            else:
                t = T[p]
            times[a] = t
            if t <= T[p]:
                cost += (T[p] - t) * PE[p]
            else:
                cost += (t - T[p]) * PL[p]
            if a > 0:
                prev = seq[a - 1]
                upper = t - float(sep[prev][p])

        return True, cost, times

    def eval_sequence_relaxed(seq):
        """Always returns a forward feasible-as-possible timing and violation amount."""
        k = len(seq)
        times = [0.0] * k
        violation = 0.0
        cost = 0.0
        for a, p in enumerate(seq):
            if a == 0:
                t = max(E[p], min(T[p], L[p]))
            else:
                prev = seq[a - 1]
                t = max(E[p], times[a - 1] + float(sep[prev][p]))
                if t < T[p]:
                    t = min(T[p], L[p])
            if t > L[p]:
                violation += t - L[p]
            times[a] = min(max(t, E[p]), max(L[p], t))
            tt = times[a]
            if tt <= T[p]:
                cost += (T[p] - tt) * PE[p]
            else:
                cost += (tt - T[p]) * PL[p]
        return violation, cost, times

    def all_times(seqs):
        out = {}
        total_cost = 0.0
        feasible = True
        for r, seq in enumerate(seqs):
            ok, c, ts = eval_sequence(seq)
            if not ok:
                feasible = False
                _, c, ts = eval_sequence_relaxed(seq)
            total_cost += c
            for idx, p in enumerate(seq):
                out[p] = (ts[idx], r)
        return feasible, total_cost, out

    # Several priority rules; keep the best complete feasible schedule.
    orders = []
    ids = list(range(n))
    orders.append(sorted(ids, key=lambda i: (L[i], T[i], E[i])))
    orders.append(sorted(ids, key=lambda i: (T[i], L[i], E[i])))
    orders.append(sorted(ids, key=lambda i: (E[i], T[i], L[i])))
    orders.append(sorted(ids, key=lambda i: ((L[i] - E[i]), T[i])))
    orders.append(sorted(ids, key=lambda i: (T[i] - E[i], L[i])))

    best_seqs = None
    best_cost = 1e100
    best_feasible = False

    for order in orders:
        if time.time() > deadline:
            break

        seqs = [[] for _ in range(m)]
        feasible_build = True

        for p in order:
            if time.time() > deadline:
                break

            best_move = None
            best_move_score = 1e100

            total_len = sum(len(s) for s in seqs)

            for r in range(m):
                seq = seqs[r]
                positions = range(len(seq) + 1)

                # For very large instances, examine a smaller set of promising insertion points.
                if total_len > 220 and len(seq) > 18:
                    pos_set = {0, len(seq)}
                    for a, q in enumerate(seq):
                        if abs(T[q] - T[p]) <= 1000 or len(pos_set) < 12:
                            pos_set.add(a)
                            pos_set.add(a + 1)
                    positions = sorted(pos_set)

                for pos in positions:
                    cand = seq[:pos] + [p] + seq[pos:]
                    ok, c, _ = eval_sequence(cand)
                    if ok:
                        # Prefer low runway cost, and mildly prefer keeping time order close to target order.
                        score = c + 0.0001 * abs(pos - len(seq) / 2.0)
                        if score < best_move_score:
                            best_move_score = score
                            best_move = (r, pos, cand)

            if best_move is not None:
                r, pos, cand = best_move
                seqs[r] = cand
            else:
                feasible_build = False
                # Fallback: append to the runway with smallest relaxed violation.
                br = 0
                bscore = 1e100
                bcand = None
                for r in range(m):
                    cand = seqs[r] + [p]
                    v, c, _ = eval_sequence_relaxed(cand)
                    score = v * 1e9 + c
                    if score < bscore:
                        bscore = score
                        br = r
                        bcand = cand
                seqs[br] = bcand

        ok, c, _ = all_times(seqs)
        if ok and c < best_cost:
            best_cost = c
            best_seqs = [s[:] for s in seqs]
            best_feasible = True
        elif best_seqs is None:
            best_seqs = [s[:] for s in seqs]

    if best_seqs is None:
        best_seqs = [[] for _ in range(m)]
        for i in range(n):
            best_seqs[i % m].append(i)

    # Local improvement by single-plane relocation/insertion.
    if best_feasible:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_ok, current_cost, _ = all_times(best_seqs)
            if not current_ok:
                break

            for r1 in range(m):
                if time.time() > deadline:
                    break
                for pos1 in range(len(best_seqs[r1])):
                    if time.time() > deadline:
                        break

                    p = best_seqs[r1][pos1]
                    base_r1 = best_seqs[r1][:pos1] + best_seqs[r1][pos1 + 1:]

                    for r2 in range(m):
                        if time.time() > deadline:
                            break

                        maxpos = len(best_seqs[r2]) + 1
                        for pos2 in range(maxpos):
                            if r1 == r2:
                                if pos2 == pos1 or pos2 == pos1 + 1:
                                    continue
                                temp = best_seqs[r1][:]
                                temp.pop(pos1)
                                ins = pos2
                                if pos2 > pos1:
                                    ins -= 1
                                temp.insert(ins, p)
                                cand_seqs = [s[:] for s in best_seqs]
                                cand_seqs[r1] = temp
                            else:
                                cand_seqs = [s[:] for s in best_seqs]
                                cand_seqs[r1] = base_r1[:]
                                cand = best_seqs[r2][:]
                                cand.insert(pos2, p)
                                cand_seqs[r2] = cand

                            ok, c, _ = all_times(cand_seqs)
                            if ok and c + 1e-7 < current_cost:
                                best_seqs = cand_seqs
                                best_cost = c
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

    feasible, cost, times_map = all_times(best_seqs)

    # If the constructed schedule is infeasible, use a conservative append-by-earliest fallback.
    if not feasible:
        seqs = [[] for _ in range(m)]
        ready = [0.0] * m
        last = [-1] * m
        for p in sorted(range(n), key=lambda i: (E[i], L[i], T[i])):
            br = 0
            bt = 1e100
            for r in range(m):
                if last[r] == -1:
                    t = E[p]
                else:
                    t = max(E[p], ready[r] + float(sep[last[r]][p]))
                if t < bt:
                    bt = t
                    br = r
            seqs[br].append(p)
            ready[br] = bt
            last[br] = p
        feasible, cost, times_map = all_times(seqs)

    schedule = {}
    for i in range(n):
        if i in times_map:
            t, r = times_map[i]
        else:
            t, r = T[i], 0
        if t < E[i]:
            t = E[i]
        if t > L[i] and feasible:
            t = L[i]
        schedule[i + 1] = {"landing_time": float(t), "runway": int(r + 1)}

    return {"schedule": schedule}