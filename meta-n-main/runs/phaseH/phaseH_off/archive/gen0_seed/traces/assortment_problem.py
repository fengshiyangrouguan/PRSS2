def solve(**kwargs):
    import time
    import math

    start_time = time.time()
    deadline = start_time + 8.5

    stocks = kwargs.get("stocks", [])
    pieces = kwargs.get("pieces", [])
    m = kwargs.get("m", len(pieces))

    EPS = 1e-9

    def piece_area(i):
        return float(pieces[i].get("length", 0.0)) * float(pieces[i].get("width", 0.0))

    def stock_area(sidx):
        return float(stocks[sidx].get("length", 0.0)) * float(stocks[sidx].get("width", 0.0))

    def orientations_for_piece(i, L, W):
        pl = float(pieces[i].get("length", 0.0))
        pw = float(pieces[i].get("width", 0.0))
        opts = []
        if pl <= L + EPS and pw <= W + EPS:
            opts.append((pl, pw, 0))
        if abs(pl - pw) > EPS and pw <= L + EPS and pl <= W + EPS:
            opts.append((pw, pl, 1))
        return opts

    def try_place_in_bin(bin_obj, pidx, stock_L, stock_W):
        opts = orientations_for_piece(pidx, stock_L, stock_W)
        if not opts:
            return False

        best = None

        # Try existing shelves first.
        for oi, (rw, rh, ori) in enumerate(opts):
            for si, sh in enumerate(bin_obj["shelves"]):
                if rh <= sh["h"] + EPS and sh["x"] + rw <= stock_L + EPS:
                    waste_height = sh["h"] - rh
                    rem_x = stock_L - (sh["x"] + rw)
                    score = (waste_height, rem_x)
                    if best is None or score < best[0]:
                        best = (score, "shelf", si, rw, rh, ori)

        if best is not None:
            _, _, si, rw, rh, ori = best
            sh = bin_obj["shelves"][si]
            x = sh["x"]
            y = sh["y"]
            sh["x"] += rw
            bin_obj["placements"].append({
                "piece": pidx + 1,
                "x": x,
                "y": y,
                "orientation": ori
            })
            bin_obj["used_area"] += rw * rh
            return True

        # Open a new shelf.
        best_new = None
        for rw, rh, ori in opts:
            if bin_obj["next_y"] + rh <= stock_W + EPS:
                rem_x = stock_L - rw
                rem_y = stock_W - (bin_obj["next_y"] + rh)
                score = (rem_y, rem_x)
                if best_new is None or score < best_new[0]:
                    best_new = (score, rw, rh, ori)

        if best_new is not None:
            _, rw, rh, ori = best_new
            y = bin_obj["next_y"]
            bin_obj["shelves"].append({"y": y, "h": rh, "x": rw})
            bin_obj["next_y"] += rh
            bin_obj["placements"].append({
                "piece": pidx + 1,
                "x": 0.0,
                "y": y,
                "orientation": ori
            })
            bin_obj["used_area"] += rw * rh
            return True

        return False

    def build_solution_for_stock_type(sidx, counts):
        stock_L = float(stocks[sidx].get("length", 0.0))
        stock_W = float(stocks[sidx].get("width", 0.0))
        sarea = stock_L * stock_W

        if stock_L <= EPS or stock_W <= EPS or sarea <= EPS:
            return None

        items = []
        for i, c in enumerate(counts):
            if c <= 0:
                continue
            if not orientations_for_piece(i, stock_L, stock_W):
                return None
            for _ in range(c):
                items.append(i)

        if not items:
            return {
                "objective": 0.0,
                "placements": {}
            }

        items.sort(key=lambda i: (max(float(pieces[i]["length"]), float(pieces[i]["width"])),
                                  piece_area(i)), reverse=True)

        bins = []

        for pidx in items:
            if time.time() > deadline:
                break

            placed = False

            # First-fit decreasing, but try bins with more free area first.
            order = list(range(len(bins)))
            order.sort(key=lambda bi: bins[bi]["used_area"], reverse=True)

            for bi in order:
                if try_place_in_bin(bins[bi], pidx, stock_L, stock_W):
                    placed = True
                    break

            if not placed:
                new_bin = {"shelves": [], "next_y": 0.0, "placements": [], "used_area": 0.0}
                if not try_place_in_bin(new_bin, pidx, stock_L, stock_W):
                    return None
                bins.append(new_bin)

        total_used = sum(b["used_area"] for b in bins)
        total_stock = len(bins) * sarea
        if total_stock <= EPS:
            obj = 0.0
        else:
            obj = max(0.0, min(1.0, (total_stock - total_used) / total_stock))

        placements = {}
        for k, b in enumerate(bins, 1):
            placements[k] = {
                "stock_type": sidx + 1,
                "placements": b["placements"]
            }

        return {
            "objective": obj,
            "placements": placements
        }

    # Minimum required counts are always safe with respect to max bounds.
    min_counts = []
    total_min = 0
    for p in pieces:
        c = int(p.get("min", 0))
        if c < 0:
            c = 0
        min_counts.append(c)
        total_min += c

    best = None

    # If no mandatory pieces exist, place one optional piece only if it improves over using stock wastefully.
    if total_min == 0:
        best_single = None
        for sidx in range(len(stocks)):
            if time.time() > deadline:
                break
            L = float(stocks[sidx].get("length", 0.0))
            W = float(stocks[sidx].get("width", 0.0))
            sa = L * W
            if sa <= EPS:
                continue
            for i, p in enumerate(pieces):
                if int(p.get("max", 0)) <= 0:
                    continue
                opts = orientations_for_piece(i, L, W)
                if not opts:
                    continue
                rw, rh, ori = opts[0]
                obj = max(0.0, min(1.0, (sa - rw * rh) / sa))
                sol = {
                    "objective": obj,
                    "placements": {
                        1: {
                            "stock_type": sidx + 1,
                            "placements": [{
                                "piece": i + 1,
                                "x": 0.0,
                                "y": 0.0,
                                "orientation": ori
                            }]
                        }
                    }
                }
                if best_single is None or obj < best_single["objective"]:
                    best_single = sol

        if best_single is not None:
            return best_single

        return {
            "objective": 0.0,
            "placements": {}
        }

    # Try packing all minimum-required pieces into each individual stock type.
    for sidx in range(len(stocks)):
        if time.time() > deadline:
            break
        sol = build_solution_for_stock_type(sidx, min_counts)
        if sol is not None:
            if best is None or sol["objective"] < best["objective"]:
                best = sol

    if best is not None:
        return best

    # Fallback: try to place any subset of mandatory pieces is invalid, but return best-effort empty
    # rather than risking malformed output. If the instance is feasible, the code above should find
    # a valid multi-stock packing for at least one stock type.
    return {
        "objective": 0.0,
        "placements": {}
    }