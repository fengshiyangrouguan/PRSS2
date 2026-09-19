def solve(**kwargs):
    import time
    import math

    start_time = time.time()
    deadline = start_time + 8.5

    stocks = kwargs.get("stocks", [])
    pieces = kwargs.get("pieces", [])
    m = kwargs.get("m", len(pieces))

    eps = 1e-9

    def area_stock(si):
        return float(stocks[si]["length"]) * float(stocks[si]["width"])

    def area_piece(pi):
        return float(pieces[pi]["length"]) * float(pieces[pi]["width"])

    def fits_piece_stock(pi, si):
        pl = float(pieces[pi]["length"])
        pw = float(pieces[pi]["width"])
        sl = float(stocks[si]["length"])
        sw = float(stocks[si]["width"])
        return (pl <= sl + eps and pw <= sw + eps) or (pw <= sl + eps and pl <= sw + eps)

    def add_free_rect(free_rects, rect):
        x, y, w, h = rect
        if w <= eps or h <= eps:
            return
        free_rects.append(rect)

    def place_in_bin(bin_obj, pi):
        sl = float(stocks[bin_obj["stock_type"]]["length"])
        sw = float(stocks[bin_obj["stock_type"]]["width"])
        pl0 = float(pieces[pi]["length"])
        pw0 = float(pieces[pi]["width"])

        best = None
        best_key = None

        for ri, rect in enumerate(bin_obj["free"]):
            rx, ry, rw, rh = rect
            for orient in (0, 1):
                if orient == 0:
                    pl, pw = pl0, pw0
                else:
                    pl, pw = pw0, pl0
                if pl <= rw + eps and pw <= rh + eps:
                    waste = rw * rh - pl * pw
                    short_side = min(rw - pl, rh - pw)
                    key = (waste, short_side, ry, rx)
                    if best is None or key < best_key:
                        best = (ri, rx, ry, rw, rh, pl, pw, orient)
                        best_key = key

        if best is None:
            return False

        ri, rx, ry, rw, rh, pl, pw, orient = best
        bin_obj["free"].pop(ri)

        rem_right = rw - pl
        rem_top = rh - pw

        if rem_right * rh >= rem_top * rw:
            add_free_rect(bin_obj["free"], (rx + pl, ry, rem_right, rh))
            add_free_rect(bin_obj["free"], (rx, ry + pw, pl, rem_top))
        else:
            add_free_rect(bin_obj["free"], (rx, ry + pw, rw, rem_top))
            add_free_rect(bin_obj["free"], (rx + pl, ry, rem_right, pw))

        bin_obj["placements"].append({
            "piece": pi + 1,
            "x": rx,
            "y": ry,
            "orientation": orient
        })
        bin_obj["used"] += pl0 * pw0
        return True

    def try_pack(counts, allowed_stock_types):
        if time.time() > deadline:
            return None

        items = []
        for pi, c in enumerate(counts):
            for _ in range(int(c)):
                items.append(pi)

        if not items:
            return {
                "objective": 0.0,
                "placements": {}
            }

        items.sort(key=lambda p: area_piece(p), reverse=True)

        bins = []

        allowed_stock_types = list(allowed_stock_types)
        allowed_stock_types.sort(key=lambda s: area_stock(s))

        for pi in items:
            if time.time() > deadline:
                return None

            placed = False

            best_existing = None
            best_existing_key = None

            for bi, b in enumerate(bins):
                tmp = {
                    "stock_type": b["stock_type"],
                    "free": list(b["free"]),
                    "placements": list(b["placements"]),
                    "used": b["used"]
                }
                if place_in_bin(tmp, pi):
                    free_area = sum(r[2] * r[3] for r in tmp["free"])
                    key = (free_area, area_stock(tmp["stock_type"]))
                    if best_existing is None or key < best_existing_key:
                        best_existing = (bi, tmp)
                        best_existing_key = key

            if best_existing is not None:
                bins[best_existing[0]] = best_existing[1]
                placed = True

            if not placed:
                best_new = None
                best_new_key = None
                for si in allowed_stock_types:
                    if not fits_piece_stock(pi, si):
                        continue
                    b = {
                        "stock_type": si,
                        "free": [(0.0, 0.0, float(stocks[si]["length"]), float(stocks[si]["width"]))],
                        "placements": [],
                        "used": 0.0
                    }
                    if place_in_bin(b, pi):
                        key = (area_stock(si) - area_piece(pi), area_stock(si))
                        if best_new is None or key < best_new_key:
                            best_new = b
                            best_new_key = key

                if best_new is None:
                    return None
                bins.append(best_new)

        total_stock_area = sum(area_stock(b["stock_type"]) for b in bins)
        total_used_area = sum(b["used"] for b in bins)

        if total_stock_area <= eps:
            obj = 0.0
        else:
            obj = max(0.0, (total_stock_area - total_used_area) / total_stock_area)

        placements = {}
        for idx, b in enumerate(bins, 1):
            placements[idx] = {
                "stock_type": b["stock_type"] + 1,
                "placements": b["placements"]
            }

        return {
            "objective": obj,
            "placements": placements
        }

    nstocks = len(stocks)

    mins = [int(pieces[i].get("min", 0)) for i in range(m)]
    maxs = [int(pieces[i].get("max", mins[i])) for i in range(m)]

    for i in range(m):
        if maxs[i] < mins[i]:
            maxs[i] = mins[i]

    if sum(maxs) == 0:
        return {
            "objective": 0.0,
            "placements": {}
        }

    stock_sets = []
    for i in range(nstocks):
        stock_sets.append((i,))
    for i in range(nstocks):
        for j in range(i + 1, nstocks):
            stock_sets.append((i, j))

    def covers_required(stock_set):
        for pi, c in enumerate(mins):
            if c > 0:
                ok = False
                for si in stock_set:
                    if fits_piece_stock(pi, si):
                        ok = True
                        break
                if not ok:
                    return False
        return True

    stock_sets = [s for s in stock_sets if covers_required(s)]
    stock_sets.sort(key=lambda ss: (len(ss), min(area_stock(s) for s in ss), sum(area_stock(s) for s in ss)))

    best = None

    count_scenarios = []

    count_scenarios.append(list(mins))

    if sum(maxs) != sum(mins):
        count_scenarios.append(list(maxs))

        mid = []
        for a, b in zip(mins, maxs):
            mid.append((a + b) // 2)
        count_scenarios.append(mid)

        area_sorted = sorted(range(m), key=lambda i: area_piece(i), reverse=True)
        greedy = list(mins)
        for pi in area_sorted:
            if maxs[pi] > greedy[pi]:
                greedy[pi] = maxs[pi]
        count_scenarios.append(greedy)

    seen_counts = set()
    unique_scenarios = []
    for c in count_scenarios:
        t = tuple(c)
        if t not in seen_counts:
            seen_counts.add(t)
            unique_scenarios.append(c)

    for counts in unique_scenarios:
        if time.time() > deadline:
            break
        for ss in stock_sets:
            if time.time() > deadline:
                break
            sol = try_pack(counts, ss)
            if sol is not None:
                if best is None or sol["objective"] < best["objective"]:
                    best = sol

    if best is None:
        required_items = []
        for pi, c in enumerate(mins):
            for _ in range(c):
                required_items.append(pi)

        placements = {}
        sid = 1
        total_stock_area = 0.0
        total_used_area = 0.0
        used_stock_types = []

        for pi in required_items:
            candidates = []
            for si in range(nstocks):
                if fits_piece_stock(pi, si):
                    candidates.append(si)
            if not candidates:
                continue
            candidates.sort(key=lambda s: area_stock(s))
            si = candidates[0]
            if si not in used_stock_types:
                if len(used_stock_types) < 2:
                    used_stock_types.append(si)
                else:
                    si = used_stock_types[0]
                    if not fits_piece_stock(pi, si):
                        si = used_stock_types[1]
            pl = float(pieces[pi]["length"])
            pw = float(pieces[pi]["width"])
            sl = float(stocks[si]["length"])
            sw = float(stocks[si]["width"])
            orient = 0
            if not (pl <= sl + eps and pw <= sw + eps):
                orient = 1

            placements[sid] = {
                "stock_type": si + 1,
                "placements": [{
                    "piece": pi + 1,
                    "x": 0.0,
                    "y": 0.0,
                    "orientation": orient
                }]
            }
            sid += 1
            total_stock_area += area_stock(si)
            total_used_area += area_piece(pi)

        obj = 0.0 if total_stock_area <= eps else max(0.0, (total_stock_area - total_used_area) / total_stock_area)
        best = {
            "objective": obj,
            "placements": placements
        }

    return best