def solve(**kwargs):
    import time
    import math

    start_time = time.time()
    deadline = start_time + 8.5

    m = kwargs.get("m", len(kwargs.get("pieces", [])))
    stocks = kwargs.get("stocks", [])
    pieces = kwargs.get("pieces", [])

    if not stocks:
        return {"objective": 0.0, "placements": {}}

    eps = 1e-9

    p_len = [float(p.get("length", 0.0)) for p in pieces]
    p_wid = [float(p.get("width", 0.0)) for p in pieces]
    p_min = [int(p.get("min", 0)) for p in pieces]
    p_max = [int(p.get("max", p_min[i] if i < len(p_min) else 0)) for i, p in enumerate(pieces)]
    p_area = [p_len[i] * p_wid[i] for i in range(m)]

    def dims_for(piece_idx, orientation):
        if orientation == 0:
            return p_len[piece_idx], p_wid[piece_idx]
        return p_wid[piece_idx], p_len[piece_idx]

    def can_fit_piece_in_stock(piece_idx, L, W):
        a, b = p_len[piece_idx], p_wid[piece_idx]
        return (a <= L + eps and b <= W + eps) or (b <= L + eps and a <= W + eps)

    def stock_area(stock_idx):
        return float(stocks[stock_idx].get("length", 0.0)) * float(stocks[stock_idx].get("width", 0.0))

    def make_solution(stock_idx, bins, counts):
        placements_out = {}
        sid = 1
        used_area = 0.0
        for b in bins:
            pls = []
            for pl in b["placements"]:
                pi = pl["piece"]
                used_area += p_area[pi]
                pls.append({
                    "piece": pi + 1,
                    "x": pl["x"],
                    "y": pl["y"],
                    "orientation": pl["orientation"]
                })
            placements_out[sid] = {
                "stock_type": stock_idx + 1,
                "placements": pls
            }
            sid += 1

        total_stock_area = len(bins) * stock_area(stock_idx)
        if total_stock_area <= eps:
            obj = 0.0
        else:
            obj = max(0.0, (total_stock_area - used_area) / total_stock_area)

        return {"objective": obj, "placements": placements_out}

    def try_place_in_existing_shelf(bin_obj, piece_idx, L, W):
        best = None
        for si, sh in enumerate(bin_obj["shelves"]):
            rem_w = L - sh["x"]
            for ori in (0, 1):
                pw, ph = dims_for(piece_idx, ori)
                if pw <= rem_w + eps and ph <= sh["h"] + eps:
                    score = (rem_w - pw, sh["h"] - ph)
                    if best is None or score < best[0]:
                        best = (score, si, ori, pw, ph)
        if best is None:
            return False
        _, si, ori, pw, ph = best
        sh = bin_obj["shelves"][si]
        x, y = sh["x"], sh["y"]
        sh["x"] += pw
        bin_obj["placements"].append({
            "piece": piece_idx,
            "x": x,
            "y": y,
            "orientation": ori
        })
        return True

    def try_open_shelf_in_bin(bin_obj, piece_idx, L, W):
        best = None
        for ori in (0, 1):
            pw, ph = dims_for(piece_idx, ori)
            if pw <= L + eps and bin_obj["y_used"] + ph <= W + eps:
                score = (ph, L - pw)
                if best is None or score < best[0]:
                    best = (score, ori, pw, ph)
        if best is None:
            return False
        _, ori, pw, ph = best
        y = bin_obj["y_used"]
        bin_obj["shelves"].append({"y": y, "h": ph, "x": pw})
        bin_obj["y_used"] += ph
        bin_obj["placements"].append({
            "piece": piece_idx,
            "x": 0.0,
            "y": y,
            "orientation": ori
        })
        return True

    def pack_mandatory_for_stock(stock_idx):
        L = float(stocks[stock_idx].get("length", 0.0))
        W = float(stocks[stock_idx].get("width", 0.0))

        for i in range(m):
            if p_min[i] > 0 and not can_fit_piece_in_stock(i, L, W):
                return None

        mandatory = []
        for i in range(m):
            for _ in range(max(0, p_min[i])):
                mandatory.append(i)

        mandatory.sort(key=lambda i: (max(p_len[i], p_wid[i]), p_area[i], min(p_len[i], p_wid[i])), reverse=True)

        bins = []
        counts = [0] * m

        for pi in mandatory:
            if time.time() > deadline:
                break

            placed = False

            bin_order = sorted(
                range(len(bins)),
                key=lambda bi: (W - bins[bi]["y_used"], len(bins[bi]["shelves"])),
                reverse=True
            )

            for bi in bin_order:
                if try_place_in_existing_shelf(bins[bi], pi, L, W):
                    placed = True
                    break

            if not placed:
                for bi in bin_order:
                    if try_open_shelf_in_bin(bins[bi], pi, L, W):
                        placed = True
                        break

            if not placed:
                b = {"shelves": [], "placements": [], "y_used": 0.0}
                if not try_open_shelf_in_bin(b, pi, L, W):
                    return None
                bins.append(b)

            counts[pi] += 1

        if any(counts[i] < p_min[i] for i in range(m)):
            return None

        return bins, counts

    def fill_optional_existing(stock_idx, bins, counts):
        L = float(stocks[stock_idx].get("length", 0.0))
        W = float(stocks[stock_idx].get("width", 0.0))

        candidates = [i for i in range(m) if counts[i] < p_max[i] and can_fit_piece_in_stock(i, L, W)]
        candidates.sort(key=lambda i: (p_area[i], max(p_len[i], p_wid[i])), reverse=True)

        improved = True
        loops = 0
        while improved and candidates and time.time() < deadline and loops < 100000:
            improved = False
            loops += 1

            candidates = [i for i in candidates if counts[i] < p_max[i]]
            candidates.sort(key=lambda i: (p_area[i], max(p_len[i], p_wid[i])), reverse=True)

            for pi in candidates:
                if counts[pi] >= p_max[pi]:
                    continue

                best = None

                for bi, b in enumerate(bins):
                    for si, sh in enumerate(b["shelves"]):
                        rem_w = L - sh["x"]
                        for ori in (0, 1):
                            pw, ph = dims_for(pi, ori)
                            if pw <= rem_w + eps and ph <= sh["h"] + eps:
                                score = (rem_w - pw, sh["h"] - ph, -p_area[pi])
                                if best is None or score < best[0]:
                                    best = (score, "shelf", bi, si, ori, pw, ph)

                    for ori in (0, 1):
                        pw, ph = dims_for(pi, ori)
                        if pw <= L + eps and b["y_used"] + ph <= W + eps:
                            score = (W - (b["y_used"] + ph), L - pw, -p_area[pi])
                            if best is None or score < best[0]:
                                best = (score, "new_shelf", bi, None, ori, pw, ph)

                if best is not None:
                    _, kind, bi, si, ori, pw, ph = best
                    b = bins[bi]
                    if kind == "shelf":
                        sh = b["shelves"][si]
                        x, y = sh["x"], sh["y"]
                        sh["x"] += pw
                    else:
                        x, y = 0.0, b["y_used"]
                        b["shelves"].append({"y": y, "h": ph, "x": pw})
                        b["y_used"] += ph

                    b["placements"].append({
                        "piece": pi,
                        "x": x,
                        "y": y,
                        "orientation": ori
                    })
                    counts[pi] += 1
                    improved = True
                    break

        return bins, counts

    def build_optional_single_stock(stock_idx):
        L = float(stocks[stock_idx].get("length", 0.0))
        W = float(stocks[stock_idx].get("width", 0.0))

        bins = [{"shelves": [], "placements": [], "y_used": 0.0}]
        counts = [0] * m

        candidates = [i for i in range(m) if p_max[i] > 0 and can_fit_piece_in_stock(i, L, W)]
        candidates.sort(key=lambda i: (p_area[i], max(p_len[i], p_wid[i])), reverse=True)

        improved = True
        loops = 0
        while improved and candidates and time.time() < deadline and loops < 100000:
            improved = False
            loops += 1
            candidates = [i for i in candidates if counts[i] < p_max[i]]
            candidates.sort(key=lambda i: (p_area[i], max(p_len[i], p_wid[i])), reverse=True)

            for pi in candidates:
                b = bins[0]
                if try_place_in_existing_shelf(b, pi, L, W) or try_open_shelf_in_bin(b, pi, L, W):
                    counts[pi] += 1
                    improved = True
                    break

        if not bins[0]["placements"]:
            return None
        return bins, counts

    mandatory_total = sum(max(0, x) for x in p_min)

    best_solution = None
    best_obj = float("inf")

    stock_indices = list(range(len(stocks)))
    stock_indices.sort(key=lambda si: stock_area(si))

    for si in stock_indices:
        if time.time() > deadline:
            break

        if stock_area(si) <= eps:
            continue

        if mandatory_total > 0:
            packed = pack_mandatory_for_stock(si)
            if packed is None:
                continue
            bins, counts = packed
            bins, counts = fill_optional_existing(si, bins, counts)
        else:
            packed = build_optional_single_stock(si)
            if packed is None:
                continue
            bins, counts = packed

        sol = make_solution(si, bins, counts)
        obj = sol["objective"]

        if obj < best_obj:
            best_obj = obj
            best_solution = sol

    if best_solution is not None:
        return best_solution

    if mandatory_total == 0:
        return {"objective": 0.0, "placements": {}}

    fallback_stock = 0
    return {
        "objective": 1.0,
        "placements": {
            1: {
                "stock_type": fallback_stock + 1,
                "placements": []
            }
        }
    }