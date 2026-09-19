def mnl_assortment(products: list, no_purchase_weight: float = 1.0, max_items: int | None = None,
                   revenue_key: str = "revenue", weight_key: str = "weight") -> dict:
    """Compute a high-quality MNL revenue-maximizing assortment.

    Args:
        products: list — products in original order. Each product may be a dict with revenue and
            preference/attraction weight fields, a tuple/list like (revenue, weight), or a numeric
            revenue value. Recognized dict alternatives include revenue/profit/price and
            weight/preference/attraction/value/v.
        no_purchase_weight: float — outside-option/no-purchase preference weight v0.
        max_items: int | None — optional maximum number of offered products. If None, no cardinality
            cap is applied.
        revenue_key: str — preferred dict key for revenue if present.
        weight_key: str — preferred dict key for MNL preference weight if present.

    Returns:
        dict — {'selected': list[int], 'expected_revenue': float}, where selected contains 1-based
        product indices in ascending order.
    """
    def fields(p):
        if isinstance(p, dict):
            r = None
            for k in (revenue_key, "revenue", "profit", "price", "margin", "r"):
                if k in p:
                    r = p[k]
                    break
            if r is None:
                r = 0.0
            v = None
            for k in (weight_key, "weight", "preference", "attraction", "value", "utility", "v"):
                if k in p:
                    v = p[k]
                    break
            if v is None:
                v = 1.0
            return float(r), float(v)
        if isinstance(p, (tuple, list)):
            if len(p) >= 2:
                return float(p[0]), float(p[1])
            if len(p) == 1:
                return float(p[0]), 1.0
        return float(p), 1.0

    items = []
    for idx, p in enumerate(products, 1):
        r, v = fields(p)
        if v > 0.0 and r > 0.0:
            items.append((idx, r, v))

    if not items:
        return {"selected": [], "expected_revenue": 0.0}

    if max_items is not None:
        try:
            max_items = int(max_items)
        except Exception:
            max_items = None
    if max_items is not None and max_items <= 0:
        return {"selected": [], "expected_revenue": 0.0}

    v0 = float(no_purchase_weight)
    if v0 < 0.0:
        v0 = 0.0

    def eval_set(sel):
        num = 0.0
        den = v0
        for _, r, v in sel:
            num += r * v
            den += v
        return num / den if den > 0.0 else 0.0

    def best_for_threshold(R):
        cand = []
        for it in items:
            _, r, v = it
            gain = v * (r - R)
            if gain > 1e-12:
                cand.append((gain, it))
        cand.sort(reverse=True, key=lambda x: x[0])
        if max_items is not None:
            cand = cand[:max_items]
        return [it for _, it in cand]

    # Parametric fixed-point iteration. For MNL, the optimal set consists of items
    # with positive contribution relative to the optimal revenue; with a cardinality
    # cap, keep the largest positive contributions.
    R = 0.0
    best_sel = []
    best_R = 0.0
    for _ in range(80):
        sel = best_for_threshold(R)
        new_R = eval_set(sel)
        if new_R > best_R + 1e-12:
            best_R = new_R
            best_sel = sel
        if abs(new_R - R) <= 1e-11 * (1.0 + abs(R)):
            break
        R = new_R

    # Safety pass over revenue thresholds catches edge cases/ties and is cheap.
    thresholds = sorted(set([0.0] + [r for _, r, _ in items]))
    for R0 in thresholds:
        sel = best_for_threshold(R0 - 1e-12)
        val = eval_set(sel)
        if val > best_R + 1e-12:
            best_R = val
            best_sel = sel

    selected = sorted(idx for idx, _, _ in best_sel)
    return {"selected": selected, "expected_revenue": float(best_R)}