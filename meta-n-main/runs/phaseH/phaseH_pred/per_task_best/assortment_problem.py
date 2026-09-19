def solve(**kwargs):
    import time
    import math

    start_time = time.time()
    deadline = start_time + 8.5

    m = int(kwargs.get("m", len(kwargs.get("pieces", []))))
    stocks = kwargs.get("stocks", [])
    pieces = kwargs.get("pieces", [])

    nstocks = len(stocks)

    def stock_area(t):
        return float(stocks[t]["length"]) * float(stocks[t]["width"])

    def piece_area(i):
        return float(pieces[i]["length"]) * float(pieces[i]["width"])

    def fits_in_stock(piece_idx, stock_idx):
        pl = float(pieces[piece_idx]["length"])
        pw = float(pieces[piece_idx]["width"])
        sl = float(stocks[stock_idx]["length"])
        sw = float(stocks[stock_idx]["width"])
        return (pl <= sl + 1e-9 and pw <= sw + 1e-9) or (pw <= sl + 1e-9 and pl <= sw + 1e-9)

    def rect_contains(a, b):
        return (a[0] <= b[0] + 1e-9 and a[1] <= b[1] + 1e-9 and
                a[0] + a[2] >= b[0] + b[2] - 1e-9 and
                a[1] + a[3] >= b[1] + b[3] - 1e-9)

    def prune_free(free_rects):
        res = []
        for i, r in enumerate(free_rects):
            if r[2] <= 1e-9 or r[3] <= 1e-9:
                continue
            contained = False
            for j, s in enumerate(free_rects):
                if i != j and rect_contains(s, r):
                    contained = True
                    break
            if not contained:
                res.append(r)
        return res

    def split_free_rect(f, px, py, pw, ph):
        fx, fy, fw, fh = f
        if px >= fx + fw - 1e-9 or px + pw <= fx + 1e-9 or py >= fy + fh - 1e-9 or py + ph <= fy + 1e-9:
            return [f]

        out = []
        if px > fx + 1e-9:
            out.append((fx, fy, px - fx, fh))
        if px + pw < fx + fw - 1e-9:
            out.append((px + pw, fy, fx + fw - (px + pw), fh))
        if py > fy + 1e-9:
            out.append((fx, fy, fw, py - fy))
        if py + ph < fy + fh - 1e-9:
            out.append((fx, py + ph, fw, fy + fh - (py + ph)))
        return out

    def clone_bins(bins):
        nb = []
        for b in bins:
            nb.append({
                "stock_type": b["stock_type"],
                "free": list(b["free"]),
                "placements": [dict(p) for p in b["placements"]],
                "used": b["used"]
            })
        return nb

    def try_place_in_bin(bin_obj, piece_idx):
        pl0 = float(pieces[piece_idx]["length"])
        pw0 = float(pieces[piece_idx]["width"])

        best = None
        for ori in (0, 1):
            if ori == 0:
                pl, pw = pl0, pw0
            else:
                pl, pw = pw0, pl0

            for k, fr in enumerate(bin_obj["free"]):
                x, y, w, h = fr
                if pl <= w + 1e-9 and pw <= h + 1e-9:
                    score = (w * h - pl * pw, min(w - pl, h - pw), y, x)
                    if best is None or score < best[0]:
                        best = (score, x, y, pl, pw, ori)

        if best is None:
            return False

        _, x, y, pl, pw, ori = best
        new_free = []
        for fr in bin_obj["free"]:
            new_free.extend(split_free_rect(fr, x, y, pl, pw))
        bin_obj["free"] = prune_free(new_free)
        bin_obj["placements"].append({
            "piece": piece_idx + 1,
            "x": float(x),
            "y": float(y),
            "orientation": int(ori)
        })
        bin_obj["used"] += pl * pw
        return True

    def add_new_bin_and_place(bins, piece_idx, allowed):
        candidates = []
        for st in allowed:
            if fits_in_stock(piece_idx, st):
                candidates.append(st)
        if not candidates:
            return False

        candidates.sort(key=lambda s: stock_area(s))
        for st in candidates:
            b = {
                "stock_type": st,
                "free": [(0.0, 0.0, float(stocks[st]["length"]), float(stocks[st]["width"]))],
                "placements": [],
                "used": 0.0
            }
            if try_place_in_bin(b, piece_idx):
                bins.append(b)
                return True
        return False

    def add_piece_to_bins(bins, piece_idx, allowed):
        order = list(range(len(bins)))
        order.sort(key=lambda k: stock_area(bins[k]["stock_type"]) - bins[k]["used"])
        for k in order:
            if try_place_in_bin(bins[k], piece_idx):
                return True
        return add_new_bin_and_place(bins, piece_idx, allowed)

    def compute_obj(bins):
        total_stock = 0.0
        total_used = 0.0
        for b in bins:
            total_stock += stock_area(b["stock_type"])
            total_used += b["used"]
        if total_stock <= 1e-12:
            return 0.0
        val = (total_stock - total_used) / total_stock
        if val < 0 and val > -1e-8:
            val = 0.0
        return float(val)

    def pack_for_allowed(allowed):
        counts = [0] * m
        items = []
        for i in range(m):
            mn = int(pieces[i].get("min", 0))
            mx = int(pieces[i].get("max", mn))
            if mn > mx:
                return None, None
            for _ in range(mn):
                items.append(i)
            counts[i] = mn

        items.sort(key=lambda i: (piece_area(i), max(float(pieces[i]["length"]), float(pieces[i]["width"]))), reverse=True)

        bins = []
        for i in items:
            if time.time() > deadline:
                break
            if not any(fits_in_stock(i, st) for st in allowed):
                return None, None
            if not add_piece_to_bins(bins, i, allowed):
                return None, None

        improved = True
        while improved and time.time() < deadline:
            improved = False
            cur_obj = compute_obj(bins)
            best_trial = None
            best_obj = cur_obj

            cand = []
            for i in range(m):
                if counts[i] < int(pieces[i].get("max", counts[i])) and any(fits_in_stock(i, st) for st in allowed):
                    cand.append(i)
            cand.sort(key=lambda i: piece_area(i), reverse=True)

            for i in cand[:max(10, min(40, len(cand)))]:
                if time.time() > deadline:
                    break
                tb = clone_bins(bins)
                if add_piece_to_bins(tb, i, allowed):
                    obj = compute_obj(tb)
                    if obj < best_obj - 1e-10:
                        best_obj = obj
                        best_trial = (tb, i)

            if best_trial is not None:
                bins, pi = best_trial
                counts[pi] += 1
                improved = True

        return bins, counts

    allowed_sets = []
    for i in range(nstocks):
        allowed_sets.append((i,))
    for i in range(nstocks):
        for j in range(i + 1, nstocks):
            allowed_sets.append((i, j))

    def allowed_score(a):
        return (len(a), sum(stock_area(x) for x in a))
    allowed_sets.sort(key=allowed_score)

    best_bins = None
    best_obj = None

    for allowed in allowed_sets:
        if time.time() > deadline:
            break

        possible = True
        for i in range(m):
            if int(pieces[i].get("min", 0)) > 0 and not any(fits_in_stock(i, st) for st in allowed):
                possible = False
                break
        if not possible:
            continue

        bins, counts = pack_for_allowed(allowed)
        if bins is None:
            continue

        obj = compute_obj(bins)
        if best_obj is None or obj < best_obj - 1e-12:
            best_obj = obj
            best_bins = bins

    if best_bins is None:
        best_bins = []
        for st in range(nstocks):
            best_bins.append({
                "stock_type": st,
                "free": [(0.0, 0.0, float(stocks[st]["length"]), float(stocks[st]["width"]))],
                "placements": [],
                "used": 0.0
            })
            break
        best_obj = compute_obj(best_bins)

    placements_out = {}
    for idx, b in enumerate(best_bins, 1):
        placements_out[idx] = {
            "stock_type": int(b["stock_type"]) + 1,
            "placements": b["placements"]
        }

    if not placements_out and nstocks > 0:
        placements_out[1] = {
            "stock_type": 1,
            "placements": []
        }
        best_obj = 1.0

    return {
        "objective": float(best_obj if best_obj is not None else 0.0),
        "placements": placements_out
    }