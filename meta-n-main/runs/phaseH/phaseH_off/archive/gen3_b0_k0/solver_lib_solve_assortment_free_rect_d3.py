def solve_assortment_free_rect(stocks: list, pieces: list, m: int, deadline: float) -> dict:
    """Solve rectangular assortment packing using free-rectangle constructive search.

    Args:
        stocks: list[dict] — stock sheet types, each with numeric 'length' and 'width'.
        pieces: list[dict] — piece types, each with numeric 'length', 'width', and optional 'min'/'max' bounds.
        m: int — number of piece types to consider from pieces.
        deadline: float — absolute time.time() cutoff; the helper stops early when reached.

    Returns:
        dict — solution with keys 'objective' and 'placements'. placements maps stock ids to
        {'stock_type': 1-based stock type, 'placements': list of piece placements}, where each
        placement has 1-based 'piece', numeric 'x'/'y', and orientation 0/1.
    """
    import time

    eps = 1e-9
    nstocks = len(stocks)
    if nstocks == 0:
        return {"objective": 0.0, "placements": {}}
    m = min(m, len(pieces))

    sl = [float(stocks[i].get("length", 0.0)) for i in range(nstocks)]
    sw = [float(stocks[i].get("width", 0.0)) for i in range(nstocks)]
    sa = [sl[i] * sw[i] for i in range(nstocks)]
    pl = [float(pieces[i].get("length", 0.0)) for i in range(m)]
    pw = [float(pieces[i].get("width", 0.0)) for i in range(m)]
    pa = [pl[i] * pw[i] for i in range(m)]
    mins = [max(0, int(pieces[i].get("min", 0))) for i in range(m)]
    maxs = [max(mins[i], int(pieces[i].get("max", mins[i]))) for i in range(m)]

    def fits_stock(pi, si):
        return ((pl[pi] <= sl[si] + eps and pw[pi] <= sw[si] + eps) or
                (pw[pi] <= sl[si] + eps and pl[pi] <= sw[si] + eps))

    def add_rect(free, rect):
        x, y, w, h = rect
        if w > eps and h > eps:
            free.append((x, y, w, h))

    def prune_free(free):
        out = []
        for i, r in enumerate(free):
            x, y, w, h = r
            contained = False
            for j, q in enumerate(free):
                if i == j:
                    continue
                x2, y2, w2, h2 = q
                if x >= x2 - eps and y >= y2 - eps and x + w <= x2 + w2 + eps and y + h <= y2 + h2 + eps:
                    contained = True
                    break
            if not contained:
                out.append(r)
        return out

    def place_one(bin_obj, pi):
        si = bin_obj["stock_type"]
        best = None
        best_key = None
        for ri, (rx, ry, rw, rh) in enumerate(bin_obj["free"]):
            for ori in (0, 1):
                w, h = (pl[pi], pw[pi]) if ori == 0 else (pw[pi], pl[pi])
                if w <= rw + eps and h <= rh + eps:
                    waste = rw * rh - w * h
                    short = min(rw - w, rh - h)
                    long = max(rw - w, rh - h)
                    key = (waste, short, long, ry, rx)
                    if best is None or key < best_key:
                        best = (ri, rx, ry, rw, rh, w, h, ori)
                        best_key = key
        if best is None:
            return False
        ri, rx, ry, rw, rh, w, h, ori = best
        bin_obj["free"].pop(ri)
        right = rw - w
        top = rh - h
        if right * rh >= top * rw:
            add_rect(bin_obj["free"], (rx + w, ry, right, rh))
            add_rect(bin_obj["free"], (rx, ry + h, w, top))
        else:
            add_rect(bin_obj["free"], (rx, ry + h, rw, top))
            add_rect(bin_obj["free"], (rx + w, ry, right, h))
        bin_obj["free"] = prune_free(bin_obj["free"])
        bin_obj["placements"].append({"piece": pi + 1, "x": rx, "y": ry, "orientation": ori})
        bin_obj["used"] += pa[pi]
        return True

    def try_pack(counts, allowed):
        if time.time() >= deadline:
            return None
        items = []
        for pi, c in enumerate(counts):
            for _ in range(int(c)):
                items.append(pi)
        if not items:
            return {"objective": 0.0, "placements": {}}
        items.sort(key=lambda i: (max(pl[i], pw[i]), pa[i], min(pl[i], pw[i])), reverse=True)

        allowed = sorted(set(allowed), key=lambda s: sa[s])
        bins = []
        for pi in items:
            if time.time() >= deadline:
                return None
            best_existing = None
            best_key = None
            for bi, b in enumerate(bins):
                tmp = {
                    "stock_type": b["stock_type"],
                    "free": list(b["free"]),
                    "placements": list(b["placements"]),
                    "used": b["used"],
                }
                if place_one(tmp, pi):
                    largest_free = 0.0
                    free_area = 0.0
                    for r in tmp["free"]:
                        a = r[2] * r[3]
                        free_area += a
                        if a > largest_free:
                            largest_free = a
                    key = (free_area, -largest_free, sa[tmp["stock_type"]])
                    if best_existing is None or key < best_key:
                        best_existing = (bi, tmp)
                        best_key = key
            if best_existing is not None:
                bins[best_existing[0]] = best_existing[1]
                continue

            best_new = None
            best_new_key = None
            for si in allowed:
                if not fits_stock(pi, si):
                    continue
                b = {
                    "stock_type": si,
                    "free": [(0.0, 0.0, sl[si], sw[si])],
                    "placements": [],
                    "used": 0.0,
                }
                if place_one(b, pi):
                    key = (sa[si] - pa[pi], sa[si])
                    if best_new is None or key < best_new_key:
                        best_new = b
                        best_new_key = key
            if best_new is None:
                return None
            bins.append(best_new)

        total_area = sum(sa[b["stock_type"]] for b in bins)
        used_area = sum(b["used"] for b in bins)
        obj = 0.0 if total_area <= eps else max(0.0, (total_area - used_area) / total_area)
        placements = {}
        for k, b in enumerate(bins, 1):
            placements[k] = {"stock_type": b["stock_type"] + 1, "placements": b["placements"]}
        return {"objective": obj, "placements": placements}

    if sum(maxs) == 0:
        return {"objective": 0.0, "placements": {}}

    stock_sets = []
    for i in range(nstocks):
        stock_sets.append((i,))
    for i in range(nstocks):
        for j in range(i + 1, nstocks):
            stock_sets.append((i, j))

    def covers(ss):
        for pi in range(m):
            if mins[pi] > 0 and not any(fits_stock(pi, si) for si in ss):
                return False
        return True

    stock_sets = [ss for ss in stock_sets if covers(ss)]
    stock_sets.sort(key=lambda ss: (len(ss), min(sa[s] for s in ss), sum(sa[s] for s in ss)))

    scenarios = [list(mins)]
    if tuple(maxs) != tuple(mins):
        scenarios.append(list(maxs))
        scenarios.append([(mins[i] + maxs[i]) // 2 for i in range(m)])

        by_area = sorted(range(m), key=lambda i: pa[i], reverse=True)
        greedy = list(mins)
        for pi in by_area:
            greedy[pi] = maxs[pi]
            scenarios.append(list(greedy))
            if len(scenarios) >= 8:
                break

        # Area-density scenario: add pieces that fit in many stocks and carry high area.
        flexible = sorted(range(m), key=lambda i: (pa[i], sum(1 for s in range(nstocks) if fits_stock(i, s))), reverse=True)
        dense = list(mins)
        for pi in flexible:
            dense[pi] = maxs[pi]
        scenarios.append(dense)

    uniq = []
    seen = set()
    for c in scenarios:
        t = tuple(c)
        if t not in seen:
            seen.add(t)
            uniq.append(c)

    best = None
    for counts in uniq:
        if time.time() >= deadline:
            break
        for ss in stock_sets:
            if time.time() >= deadline:
                break
            sol = try_pack(counts, ss)
            if sol is not None and (best is None or sol["objective"] < best["objective"]):
                best = sol

    if best is not None:
        return best

    # Safe fallback: one required piece per smallest fitting stock sheet.
    placements = {}
    sid = 1
    total = 0.0
    used = 0.0
    for pi in range(m):
        for _ in range(mins[pi]):
            choices = [si for si in range(nstocks) if fits_stock(pi, si)]
            if not choices:
                continue
            si = min(choices, key=lambda s: sa[s])
            ori = 0
            if not (pl[pi] <= sl[si] + eps and pw[pi] <= sw[si] + eps):
                ori = 1
            placements[sid] = {
                "stock_type": si + 1,
                "placements": [{"piece": pi + 1, "x": 0.0, "y": 0.0, "orientation": ori}],
            }
            sid += 1
            total += sa[si]
            used += pa[pi]
    obj = 0.0 if total <= eps else max(0.0, (total - used) / total)
    return {"objective": obj, "placements": placements}