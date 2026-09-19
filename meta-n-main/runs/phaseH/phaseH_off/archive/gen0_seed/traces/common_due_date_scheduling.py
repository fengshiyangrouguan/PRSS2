def solve(**kwargs):
    """
    Solves the restricted single-machine common due date scheduling problem.
    Returns {'schedule': [...]} with 1-based job indices.
    """
    import time
    import random
    import math

    jobs = kwargs.get("jobs", [])
    h = kwargs.get("h", 0.6)
    n = len(jobs)

    if n == 0:
        return {"schedule": []}
    if n == 1:
        return {"schedule": [1]}

    p = [int(x[0]) for x in jobs]
    a = [int(x[1]) for x in jobs]
    b = [int(x[2]) for x in jobs]

    total_p = sum(p)
    d = int(math.floor(total_p * h))

    start_time = time.time()
    deadline = start_time + 9.15

    def pen_job(idx, c):
        if c < d:
            return a[idx] * (d - c)
        if c > d:
            return b[idx] * (c - d)
        return 0

    def cost(seq):
        c = 0
        val = 0
        for idx in seq:
            c += p[idx]
            if c < d:
                val += a[idx] * (d - c)
            elif c > d:
                val += b[idx] * (c - d)
        return val

    def adjacent_descent(seq, val):
        """Fast adjacent-swap local search; exact delta because only swapped jobs change."""
        m = len(seq)
        improved = True
        passes = 0
        while improved and time.time() < deadline:
            improved = False
            passes += 1
            c_before = 0
            i = 0
            while i < m - 1:
                if time.time() >= deadline:
                    return seq, val

                x = seq[i]
                y = seq[i + 1]

                cx_old = c_before + p[x]
                cy_old = cx_old + p[y]
                old = pen_job(x, cx_old) + pen_job(y, cy_old)

                cy_new = c_before + p[y]
                cx_new = cy_new + p[x]
                new = pen_job(y, cy_new) + pen_job(x, cx_new)

                if new < old:
                    seq[i], seq[i + 1] = seq[i + 1], seq[i]
                    val += new - old
                    improved = True
                    # after swap, completion time after position i+1 is unchanged
                    # try moving the swapped job further left
                    if i > 0:
                        c_before -= p[seq[i - 1]]
                        i -= 1
                    else:
                        c_before = 0
                        i = 0
                else:
                    c_before += p[x]
                    i += 1

            if passes >= 50:
                break
        return seq, val

    def try_candidate(seq):
        nonlocal best_seq, best_val
        if time.time() >= deadline:
            return
        val = cost(seq)
        seq, val = adjacent_descent(list(seq), val)
        if val < best_val:
            best_seq = list(seq)
            best_val = val

    indices = list(range(n))

    best_seq = indices[:]
    best_val = cost(best_seq)

    # Basic candidates
    candidate_lists = []

    candidate_lists.append(indices[:])
    candidate_lists.append(sorted(indices, key=lambda i: p[i]))
    candidate_lists.append(sorted(indices, key=lambda i: -p[i]))
    candidate_lists.append(sorted(indices, key=lambda i: (b[i] / p[i] if p[i] else b[i])))
    candidate_lists.append(sorted(indices, key=lambda i: -(a[i] / p[i] if p[i] else a[i])))

    # V-shaped schedules:
    # early side: decreasing a/p, tardy side: increasing b/p.
    # choose early set by score b-a or b/p-a/p; jobs with large tardiness penalty tend to be early.
    score_orders = [
        sorted(indices, key=lambda i: (b[i] - a[i], b[i] / p[i] if p[i] else b[i]), reverse=True),
        sorted(indices, key=lambda i: ((b[i] / p[i] if p[i] else b[i]) - (a[i] / p[i] if p[i] else a[i])), reverse=True),
        sorted(indices, key=lambda i: b[i] / p[i] if p[i] else b[i], reverse=True),
        sorted(indices, key=lambda i: -a[i] / p[i] if p[i] else -a[i], reverse=True),
    ]

    split_fracs = [0.35, 0.45, 0.55, 0.65, 0.75, h]
    for order in score_orders:
        pref_p = [0]
        for i in order:
            pref_p.append(pref_p[-1] + p[i])

        split_positions = set()
        for frac in split_fracs:
            target = int(total_p * frac)
            k = 0
            while k < n and pref_p[k] < target:
                k += 1
            for kk in (k - 2, k - 1, k, k + 1, k + 2):
                if 0 <= kk <= n:
                    split_positions.add(kk)

        k = 0
        while k < n and pref_p[k] <= d:
            k += 1
        for kk in (k - 3, k - 2, k - 1, k, k + 1, k + 2, k + 3):
            if 0 <= kk <= n:
                split_positions.add(kk)

        for k in split_positions:
            early = order[:k]
            tardy = order[k:]
            early = sorted(early, key=lambda i: (a[i] / p[i] if p[i] else a[i]), reverse=True)
            tardy = sorted(tardy, key=lambda i: (b[i] / p[i] if p[i] else b[i]))
            candidate_lists.append(early + tardy)

    # Greedy fill early side up to due date using high b/a desirability
    for keyfun in (
        lambda i: (b[i] + 1) / (a[i] + 1),
        lambda i: (b[i] + 1) / (p[i] + 1),
        lambda i: (b[i] - a[i]) / (p[i] + 1),
    ):
        order = sorted(indices, key=keyfun, reverse=True)
        early = []
        tardy = []
        used_p = 0
        for i in order:
            if used_p + p[i] <= d or used_p < d * 0.85:
                early.append(i)
                used_p += p[i]
            else:
                tardy.append(i)
        early.sort(key=lambda i: (a[i] / p[i] if p[i] else a[i]), reverse=True)
        tardy.sort(key=lambda i: (b[i] / p[i] if p[i] else b[i]))
        candidate_lists.append(early + tardy)

    for seq in candidate_lists:
        if time.time() >= deadline:
            break
        try_candidate(seq)

    # NEH-style insertion construction for moderate n
    if n <= 90 and time.time() < deadline:
        order = sorted(indices, key=lambda i: p[i] * (a[i] + b[i]), reverse=True)
        seq = []
        for job in order:
            if time.time() >= deadline:
                break
            best_local = None
            best_local_val = None
            m = len(seq)
            for pos in range(m + 1):
                trial = seq[:pos] + [job] + seq[pos:]
                v = cost(trial)
                if best_local_val is None or v < best_local_val:
                    best_local_val = v
                    best_local = trial
            seq = best_local
        if len(seq) == n:
            try_candidate(seq)

    # Randomized improvement: swap / insertion, bounded by deadline.
    rng = random.Random(1234567 + n + total_p)
    cur = best_seq[:]
    cur_val = best_val
    no_imp = 0

    while time.time() < deadline:
        if n <= 1:
            break

        if no_imp > 500:
            cur = best_seq[:]
            cur_val = best_val
            no_imp = 0

        if rng.random() < 0.55:
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            trial = cur[:]
            trial[i], trial[j] = trial[j], trial[i]
        else:
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i == j:
                continue
            trial = cur[:]
            job = trial.pop(i)
            trial.insert(j, job)

        v = cost(trial)
        if v < cur_val or rng.random() < 0.001:
            cur = trial
            cur_val = v
            cur, cur_val = adjacent_descent(cur, cur_val)
            no_imp = 0
            if cur_val < best_val:
                best_seq = cur[:]
                best_val = cur_val
        else:
            no_imp += 1

    return {"schedule": [i + 1 for i in best_seq]}