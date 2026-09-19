def segmented_mnl_assortment(revenues: list, weights: list, segment_probs: list | None = None,
                             no_purchase_weights: list | float = 1.0,
                             max_items: int | None = None, budgets: list | None = None,
                             budget_limit: float | None = None,
                             time_limit: float = 2.0) -> dict:
    """Optimize a multi-segment MNL assortment by greedy construction and local search.

    Args:
        revenues: list[float] — product revenues/profits/prices, one per product.
        weights: list — segment-product attraction weights. Expected shape is either
            list[list[float]] with weights[s][i], or a single list[float] for one segment.
        segment_probs: list[float] | None — nonnegative segment arrival probabilities/weights.
            If None, all segments are weighted equally.
        no_purchase_weights: list[float] | float — outside-option weights v0_s, either one
            scalar for all segments or one value per segment.
        max_items: int | None — optional maximum number of offered products.
        budgets: list[float] | None — optional product costs/space usages for a budget constraint.
        budget_limit: float | None — optional maximum total budget/cost for selected products.
        time_limit: float — approximate local-search time limit in seconds.

    Returns:
        dict — {'selected': list[int], 'expected_revenue': float}, where selected contains
        1-based product indices in ascending order and expected_revenue is the segmented MNL value.
    """
    import time
    import random

    start = time.time()
    r = [float(x) for x in revenues]
    n = len(r)
    if n == 0:
        return {"selected": [], "expected_revenue": 0.0}

    if not weights:
        weights = [[1.0] * n]
    elif all(not isinstance(x, (list, tuple)) for x in weights):
        weights = [weights]
    W = []
    for row in weights:
        rr = [float(x) for x in row[:n]]
        if len(rr) < n:
            rr += [0.0] * (n - len(rr))
        W.append(rr)
    m = len(W)

    if segment_probs is None:
        prob = [1.0 / m] * m
    else:
        prob = [max(0.0, float(x)) for x in segment_probs[:m]]
        if len(prob) < m:
            prob += [0.0] * (m - len(prob))
        sprob = sum(prob)
        prob = [x / sprob for x in prob] if sprob > 0 else [1.0 / m] * m

    if isinstance(no_purchase_weights, (list, tuple)):
        v0 = [max(0.0, float(x)) for x in no_purchase_weights[:m]]
        if len(v0) < m:
            v0 += [1.0] * (m - len(v0))
    else:
        v0 = [max(0.0, float(no_purchase_weights))] * m

    if max_items is not None:
        try:
            max_items = int(max_items)
        except Exception:
            max_items = None
    if max_items is not None and max_items <= 0:
        return {"selected": [], "expected_revenue": 0.0}

    cost = None
    if budgets is not None and budget_limit is not None:
        cost = [float(x) for x in budgets[:n]]
        if len(cost) < n:
            cost += [0.0] * (n - len(cost))
        budget_limit = float(budget_limit)

    def feasible(sel):
        if max_items is not None and len(sel) > max_items:
            return False
        if cost is not None and sum(cost[i] for i in sel) > budget_limit + 1e-9:
            return False
        return True

    def value(sel):
        if not sel:
            return 0.0
        total = 0.0
        for s in range(m):
            den = v0[s]
            num = 0.0
            row = W[s]
            for i in sel:
                vi = row[i]
                if vi > 0:
                    den += vi
                    num += vi * r[i]
            if den > 0:
                total += prob[s] * num / den
        return total

    # Candidate starts: empty, high revenue, high average attraction*revenue, and averaged MNL helper if available.
    starts = [set()]
    avgw = [sum(prob[s] * W[s][i] for s in range(m)) for i in range(n)]
    orderings = [
        sorted(range(n), key=lambda i: r[i], reverse=True),
        sorted(range(n), key=lambda i: avgw[i] * r[i], reverse=True),
        sorted(range(n), key=lambda i: r[i] / (1.0 + (cost[i] if cost is not None else 0.0)), reverse=True),
    ]
    for order in orderings:
        sel = set()
        for i in order:
            trial = set(sel)
            trial.add(i)
            if feasible(trial):
                sel = trial
        starts.append(sel)

    try:
        avg_products = [(r[i], avgw[i]) for i in range(n)]
        cap = max_items
        base = mnl_assortment(avg_products, 1.0, cap)
        sel = set(int(i) - 1 for i in base.get("selected", []) if 1 <= int(i) <= n)
        if feasible(sel):
            starts.append(sel)
    except Exception:
        pass

    best = set()
    best_val = -1.0
    for st in starts:
        cur = set(st)
        cur_val = value(cur)
        improved = True
        while improved and time.time() - start < time_limit:
            improved = False
            # best add/drop/swap first improvement rounds
            candidates = []
            for i in range(n):
                if i not in cur:
                    candidates.append(("add", i, -1))
                else:
                    candidates.append(("drop", i, -1))
            if len(cur) > 0 and (max_items is None or len(cur) >= max_items or cost is not None):
                out_items = list(cur)
                in_items = [i for i in range(n) if i not in cur]
                for a in out_items:
                    for b in in_items[:]:
                        candidates.append(("swap", a, b))
            random.shuffle(candidates)
            local_best = cur
            local_val = cur_val
            for typ, a, b in candidates:
                if time.time() - start >= time_limit:
                    break
                trial = set(cur)
                if typ == "add":
                    trial.add(a)
                elif typ == "drop":
                    trial.remove(a)
                else:
                    trial.remove(a)
                    trial.add(b)
                if not feasible(trial):
                    continue
                tv = value(trial)
                if tv > local_val + 1e-12:
                    local_val = tv
                    local_best = trial
            if local_val > cur_val + 1e-12:
                cur, cur_val = local_best, local_val
                improved = True
        if cur_val > best_val:
            best, best_val = cur, cur_val

    return {"selected": [i + 1 for i in sorted(best)], "expected_revenue": float(max(0.0, best_val))}