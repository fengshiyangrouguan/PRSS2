def solve(**kwargs):
    import time
    import math

    start_time = time.time()
    deadline = start_time + 9.2

    stocks = kwargs.get("stocks", [])
    pieces = kwargs.get("pieces", [])
    m = kwargs.get("m", len(pieces))
    if not m:
        m = len(pieces)

    eps = 1e-9

    def p_area(i):
        return float(pieces[i]["length"]) * float(pieces[i]["width"])

    def s_area(sidx):
        return float(stocks[sidx]["length"]) * float(stocks[sidx]["width"])

    def can_fit_piece_stock(pi, si):
        pl, pw = float(pieces[pi]["length"]), float(pieces[pi]["width"])
        sl, sw = float(stocks[si]["length"]), float(stocks[si]["width"])
        return (pl <= sl + eps and pw <= sw + eps) or (pw <= sl + eps and pl <= sw + eps)

    def prune_free_rects(rects):
        out = []
        for r in rects:
            x, y, w, h = r
            if w <= eps or h <= eps:
                continue
            contained = False
            for q in rects:
                if q is r:
                    continue
                x2, y2, w2, h2 = q
                if (x >= x2 - eps and y >= y2 - eps and
                    x + w <= x2 + w2 + eps and y + h <= y2 + h2 + eps and
                    (w * h < w2 * h2 - eps or (x, y, w, h) != (x2, y2, w2, h2))):
                    contained = True
                    break
            if not contained:
                out.append(r)
        return out

    def try_place_in_bin(bin_obj, pi):
        pl, pw = float(pieces[pi]["length"]), float(pieces[pi]["width"])
        best = None
        best_score = None

        for ri, fr in enumerate(bin_obj["free"]):
            x, y, fw, fh = fr
            orientations = [(pl, pw, 0)]
            if abs(pl - pw) > eps:
                orientations.append((pw, pl, 1))
            for ww, hh, ori in orientations:
                if ww <= fw + eps and hh <= fh + eps:
                    leftover_area = fw * fh - ww * hh
                    short_side = min(fw - ww, fh - hh)
                    long_side = max(fw - ww, fh - hh)
                    score = (leftover_area, short_side, long_side)
                    if best_score is None or score < best_score:
                        best_score = score
                        best = (ri, x, y, fw, fh, ww, hh, ori)

        if best is None:
            return False

        ri, x, y, fw, fh, ww, hh, ori = best
        bin_obj["free"].pop(ri)

        # Simple guillotine split: right strip beside the item and top strip above it.
        if fw - ww > eps:
            bin_obj["free"].append((x + ww, y, fw - ww, hh))
        if fh - hh > eps:
            bin_obj["free"].append((x, y + hh, fw, fh - hh))

        bin_obj["free"] = prune_free_rects(bin_obj["free"])
        bin_obj["placements"].append({
            "piece": pi + 1,
            "x": x,
            "y": y,
            "orientation": ori
        })
        bin_obj["used_area"] += p_area(pi)
        return True

    def make_bin(si):
        return {
            "stock_type": si + 1,
            "stock_idx": si,
            "L": float(stocks[si]["length"]),
            "W": float(stocks[si]["width"]),
            "free": [(0.0, 0.0, float(stocks[si]["length"]), float(stocks[si]["width"]))],
            "placements": [],
            "used_area": 0.0
        }

    def pack_for_stock_set(stock_set, mandatory_only=False):
        if time.time() > deadline:
            return None

        counts = [0] * m
        bins = []

        mandatory_items = []
        for i in range(m):
            mn = int(pieces[i].get("min", 0))
            mx = int(pieces[i].get("max", mn))
            if mn > mx:
                return None
            for _ in range(mn):
                mandatory_items.append(i)

        mandatory_items.sort(key=lambda i: (-p_area(i), -max(float(pieces[i]["length"]), float(pieces[i]["width"]))))

        for pi in mandatory_items:
            if time.time() > deadline:
                return None

            placed = False

            # Try existing bins first.
            best_bin_order = sorted(
                range(len(bins)),
                key=lambda bi: (s_area(bins[bi]["stock_idx"]) - bins[bi]["used_area"])
            )
            for bi in best_bin_order:
                if try_place_in_bin(bins[bi], pi):
                    counts[pi] += 1
                    placed = True
                    break

            if placed:
                continue

            # Open the smallest selected stock that can fit this piece.
            candidate_stocks = [si for si in stock_set if can_fit_piece_stock(pi, si)]
            if not candidate_stocks:
                return None
            candidate_stocks.sort(key=lambda si: s_area(si))

            opened = False
            for si in candidate_stocks:
                b = make_bin(si)
                if try_place_in_bin(b, pi):
                    bins.append(b)
                    counts[pi] += 1
                    opened = True
                    break
            if not opened:
                return None

        # If no mandatory items, open one promising sheet and fill with optional pieces.
        if not bins and not mandatory_only:
            best_seed = None
            for si in stock_set:
                for pi in range(m):
                    if int(pieces[pi].get("max", 0)) > 0 and can_fit_piece_stock(pi, si):
                        ratio = p_area(pi) / s_area(si)
                        cand = (ratio, si)
                        if best_seed is None or cand > best_seed:
                            best_seed = cand
            if best_seed is not None:
                bins.append(make_bin(best_seed[1]))

        if not mandatory_only:
            # Optional fill pass: add pieces up to max into already-open sheets only.
            improved = True
            while improved and time.time() <= deadline:
                improved = False
                candidates = [
                    i for i in range(m)
                    if counts[i] < int(pieces[i].get("max", pieces[i].get("min", 0)))
                ]
                candidates.sort(key=lambda i: -p_area(i))

                best_move = None
                best_score = None

                for bi, b in enumerate(bins):
                    for pi in candidates:
                        pl, pw = float(pieces[pi]["length"]), float(pieces[pi]["width"])
                        for fr in b["free"]:
                            x, y, fw, fh = fr
                            opts = [(pl, pw), (pw, pl)] if abs(pl - pw) > eps else [(pl, pw)]
                            for ww, hh in opts:
                                if ww <= fw + eps and hh <= fh + eps:
                                    # Prefer large area and tight fit.
                                    score = (-p_area(pi), min(fw - ww, fh - hh), fw * fh - ww * hh)
                                    if best_score is None or score < best_score:
                                        best_score = score
                                        best_move = (bi, pi)
                                    break

                if best_move is not None:
                    bi, pi = best_move
                    if try_place_in_bin(bins[bi], pi):
                        counts[pi] += 1
                        improved = True

        # Validate counts.
        for i in range(m):
            if counts[i] < int(pieces[i].get("min", 0)) or counts[i] > int(pieces[i].get("max", pieces[i].get("min", 0))):
                return None

        if not bins:
            return {
                "objective": 0.0,
                "placements": {}
            }

        total_stock_area = sum(s_area(b["stock_idx"]) for b in bins)
        total_used_area = sum(b["used_area"] for b in bins)
        if total_stock_area <= eps:
            return None

        objective = max(0.0, (total_stock_area - total_used_area) / total_stock_area)

        placements = {}
        for idx, b in enumerate(bins, 1):
            placements[idx] = {
                "stock_type": b["stock_type"],
                "placements": b["placements"]
            }

        return {
            "objective": objective,
            "placements": placements
        }

    best = None

    # Try each single stock type.
    for si in range(len(stocks)):
        if time.time() > deadline:
            break
        sol = pack_for_stock_set([si])
        if sol is not None and (best is None or sol["objective"] < best["objective"]):
            best = sol

    # Try pairs of stock types, useful when different pieces fit different sheets.
    nstocks = len(stocks)
    for a in range(nstocks):
        if time.time() > deadline:
            break
        for b in range(a + 1, nstocks):
            if time.time() > deadline:
                break
            sol = pack_for_stock_set([a, b])
            if sol is not None and (best is None or sol["objective"] < best["objective"]):
                best = sol

    if best is not None:
        return best

    # Last-resort fallback: return an empty solution. This is valid only when all minimums are zero.
    all_min_zero = True
    for p in pieces:
        if int(p.get("min", 0)) > 0:
            all_min_zero = False
            break

    if all_min_zero:
        return {
            "objective": 0.0,
            "placements": {}
        }

    # If mandatory pieces could not be packed by the heuristic, still return correctly formatted output.
    return {
        "objective": 1.0,
        "placements": {}
    }