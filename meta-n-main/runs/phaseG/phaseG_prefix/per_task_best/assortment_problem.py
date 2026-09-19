def solve(**kwargs):
    import time
    import math

    start_time = time.time()
    deadline = start_time + 8.5

    stocks = kwargs.get("stocks", [])
    pieces = kwargs.get("pieces", [])
    m = kwargs.get("m", len(pieces))

    eps = 1e-9

    if not stocks:
        return {"objective": 0.0, "placements": {}}

    mins = [int(p.get("min", 0)) for p in pieces]
    maxs = [int(p.get("max", mins[i])) for i, p in enumerate(pieces)]

    total_min = sum(mins)
    if total_min == 0:
        return {"objective": 0.0, "placements": {}}

    piece_area = [float(p["length"]) * float(p["width"]) for p in pieces]

    def orientations_for_piece(pi, stock_type):
        p = pieces[pi]
        s = stocks[stock_type]
        pl, pw = float(p["length"]), float(p["width"])
        sl, sw = float(s["length"]), float(s["width"])
        res = []
        if pl <= sl + eps and pw <= sw + eps:
            res.append((pl, pw, 0))
        if pw <= sl + eps and pl <= sw + eps and (abs(pl - pw) > eps):
            res.append((pw, pl, 1))
        elif pw <= sl + eps and pl <= sw + eps:
            res.append((pw, pl, 1))
        return res

    def can_fit_piece(pi, stock_type):
        return len(orientations_for_piece(pi, stock_type)) > 0

    def add_item_to_bins(bins, pi, stock_type, allow_new_bin):
        s = stocks[stock_type]
        W, H = float(s["length"]), float(s["width"])
        ors = orientations_for_piece(pi, stock_type)
        if not ors:
            return False

        best = None

        # Existing shelves.
        for bi, b in enumerate(bins):
            for shi, sh in enumerate(b["shelves"]):
                rem = W - sh["used"]
                for w, h, ori in ors:
                    if w <= rem + eps and h <= sh["height"] + eps:
                        waste_height = sh["height"] - h
                        score = (0, waste_height, rem - w)
                        if best is None or score < best[0]:
                            best = (score, bi, shi, w, h, ori, "shelf")

        # New shelf in existing bins.
        for bi, b in enumerate(bins):
            y = b["height_used"]
            rem_h = H - y
            for w, h, ori in ors:
                if w <= W + eps and h <= rem_h + eps:
                    score = (1, rem_h - h, W - w)
                    if best is None or score < best[0]:
                        best = (score, bi, None, w, h, ori, "new_shelf")

        # New bin.
        if allow_new_bin:
            for w, h, ori in ors:
                if w <= W + eps and h <= H + eps:
                    score = (2, H - h, W - w)
                    if best is None or score < best[0]:
                        best = (score, None, None, w, h, ori, "new_bin")

        if best is None:
            return False

        _, bi, shi, w, h, ori, mode = best
        if mode == "shelf":
            b = bins[bi]
            sh = b["shelves"][shi]
            x = sh["used"]
            y = sh["y"]
            sh["used"] += w
            b["placements"].append({"piece": pi + 1, "x": x, "y": y, "orientation": ori})
            b["used_area"] += piece_area[pi]
            return True

        if mode == "new_shelf":
            b = bins[bi]
            y = b["height_used"]
            b["shelves"].append({"y": y, "height": h, "used": w})
            b["height_used"] += h
            b["placements"].append({"piece": pi + 1, "x": 0.0, "y": y, "orientation": ori})
            b["used_area"] += piece_area[pi]
            return True

        b = {
            "stock_type": stock_type,
            "shelves": [{"y": 0.0, "height": h, "used": w}],
            "height_used": h,
            "placements": [{"piece": pi + 1, "x": 0.0, "y": 0.0, "orientation": ori}],
            "used_area": piece_area[pi],
        }
        bins.append(b)
        return True

    def pack_items(stock_type, item_indices):
        bins = []
        ordered = sorted(
            item_indices,
            key=lambda i: (
                -piece_area[i],
                -max(float(pieces[i]["length"]), float(pieces[i]["width"])),
            ),
        )
        for pi in ordered:
            if time.time() > deadline:
                return None
            if not add_item_to_bins(bins, pi, stock_type, True):
                return None
        return bins

    def build_solution_for_subset(subset):
        # Assign each piece type to the feasible stock in subset with smallest stock area.
        assigned = {st: [] for st in subset}
        for pi in range(m):
            if mins[pi] <= 0:
                continue
            feasible = [st for st in subset if can_fit_piece(pi, st)]
            if not feasible:
                return None
            feasible.sort(key=lambda st: float(stocks[st]["length"]) * float(stocks[st]["width"]))
            assigned[feasible[0]].extend([pi] * mins[pi])

        all_bins = []
        for st in subset:
            if assigned[st]:
                bs = pack_items(st, assigned[st])
                if bs is None:
                    return None
                all_bins.extend(bs)

        if not all_bins:
            return {"objective": 0.0, "placements": {}}

        counts = mins[:]

        # Try to improve by adding optional pieces into already-open stock rectangles only.
        improved = True
        candidates = sorted(range(m), key=lambda i: -piece_area[i])
        while improved and time.time() < deadline:
            improved = False
            for pi in candidates:
                if counts[pi] >= maxs[pi]:
                    continue
                # Adding to existing stock always reduces waste if it fits.
                bin_order = sorted(
                    range(len(all_bins)),
                    key=lambda bi: float(stocks[all_bins[bi]["stock_type"]]["length"]) *
                                   float(stocks[all_bins[bi]["stock_type"]]["width"]) -
                                   all_bins[bi]["used_area"],
                )
                for bi in bin_order:
                    if time.time() > deadline:
                        break
                    st = all_bins[bi]["stock_type"]
                    before_len = len(all_bins)
                    target = [all_bins[bi]]
                    if add_item_to_bins(target, pi, st, False):
                        counts[pi] += 1
                        improved = True
                        break
                    # target contains the same object; no restoration needed on failure.
                    if len(all_bins) != before_len:
                        break

        total_stock_area = 0.0
        total_used_area = 0.0
        placements = {}
        sid = 1
        for b in all_bins:
            st = b["stock_type"]
            total_stock_area += float(stocks[st]["length"]) * float(stocks[st]["width"])
            total_used_area += b["used_area"]
            placements[sid] = {
                "stock_type": st + 1,
                "placements": b["placements"],
            }
            sid += 1

        if total_stock_area <= eps:
            obj = 0.0
        else:
            obj = max(0.0, (total_stock_area - total_used_area) / total_stock_area)

        return {"objective": obj, "placements": placements}

    stock_indices = list(range(len(stocks)))
    subsets = []

    # Prefer smaller-area stocks first, but test all singles and pairs if time allows.
    stock_indices.sort(key=lambda st: float(stocks[st]["length"]) * float(stocks[st]["width"]))
    for st in stock_indices:
        subsets.append((st,))
    for i in range(len(stock_indices)):
        for j in range(i + 1, len(stock_indices)):
            subsets.append((stock_indices[i], stock_indices[j]))

    best = None
    for subset in subsets:
        if time.time() > deadline:
            break
        sol = build_solution_for_subset(subset)
        if sol is not None:
            if best is None or sol["objective"] < best["objective"]:
                best = sol

    if best is not None:
        return best

    # Last-resort fallback: place each required piece in its own feasible stock.
    placements = {}
    sid = 1
    total_stock_area = 0.0
    total_used_area = 0.0
    used_stock_types = set()

    for pi in range(m):
        for _ in range(mins[pi]):
            feasible = [st for st in range(len(stocks)) if can_fit_piece(pi, st)]
            if not feasible:
                continue
            # Respect at most two stock types if possible.
            feasible.sort(key=lambda st: (
                0 if st in used_stock_types or len(used_stock_types) < 2 else 1,
                float(stocks[st]["length"]) * float(stocks[st]["width"])
            ))
            st = feasible[0]
            if st not in used_stock_types and len(used_stock_types) >= 2:
                alternatives = [x for x in feasible if x in used_stock_types]
                if alternatives:
                    st = alternatives[0]
            used_stock_types.add(st)
            w, h, ori = orientations_for_piece(pi, st)[0]
            placements[sid] = {
                "stock_type": st + 1,
                "placements": [{"piece": pi + 1, "x": 0.0, "y": 0.0, "orientation": ori}],
            }
            total_stock_area += float(stocks[st]["length"]) * float(stocks[st]["width"])
            total_used_area += piece_area[pi]
            sid += 1

    obj = 0.0 if total_stock_area <= eps else max(0.0, (total_stock_area - total_used_area) / total_stock_area)
    return {"objective": obj, "placements": placements}