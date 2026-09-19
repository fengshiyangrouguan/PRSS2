def solve(**kwargs):
    import time
    import math
    import random

    start_time = time.time()
    deadline = start_time + 8.5

    stocks = kwargs.get("stocks", [])
    pieces = kwargs.get("pieces", [])
    m = kwargs.get("m", len(pieces))

    if not stocks or not pieces:
        return {"objective": 0.0, "placements": {}}

    n_stocks = len(stocks)
    stock_area = [float(s["length"]) * float(s["width"]) for s in stocks]

    p_len = [float(p["length"]) for p in pieces]
    p_wid = [float(p["width"]) for p in pieces]
    p_area = [p_len[i] * p_wid[i] for i in range(m)]
    p_min = [int(p.get("min", 0)) for p in pieces]
    p_max = [int(p.get("max", p_min[i])) for i, p in enumerate(pieces)]

    def fits_piece_stock(pi, si):
        L = float(stocks[si]["length"])
        W = float(stocks[si]["width"])
        a = p_len[pi]
        b = p_wid[pi]
        return (a <= L + 1e-9 and b <= W + 1e-9) or (b <= L + 1e-9 and a <= W + 1e-9)

    if sum(p_min) == 0:
        return {"objective": 0.0, "placements": {}}

    # Ensure each mandatory piece type can fit somewhere.
    for i in range(m):
        if p_min[i] > 0 and not any(fits_piece_stock(i, s) for s in range(n_stocks)):
            return {"objective": 1.0, "placements": {}}

    def orient_dims(pi, si, prefer_wide=False):
        L = float(stocks[si]["length"])
        W = float(stocks[si]["width"])
        a = p_len[pi]
        b = p_wid[pi]
        opts = []
        if a <= L + 1e-9 and b <= W + 1e-9:
            opts.append((a, b, 0))
        if b <= L + 1e-9 and a <= W + 1e-9:
            opts.append((b, a, 1))
        if not opts:
            return None
        if prefer_wide:
            opts.sort(key=lambda t: (-t[0], t[1]))
        else:
            opts.sort(key=lambda t: (-t[1], t[0]))
        return opts[0]

    def pack_group(si, item_types):
        """
        Packs a list of piece type indices into stock type si using a simple shelf FFD heuristic.
        Returns list of stock-instance placement lists, or None.
        """
        if not item_types:
            return []

        L = float(stocks[si]["length"])
        W = float(stocks[si]["width"])

        # Sort large/tall pieces first.
        enriched = []
        for pi in item_types:
            od = orient_dims(pi, si)
            if od is None:
                return None
            a, b, o = od
            enriched.append((max(a, b), p_area[pi], pi))
        enriched.sort(reverse=True)

        bins = []

        for _, __, pi in enriched:
            if time.time() > deadline:
                return None

            placed = False

            # Try both orientations where feasible, generally preferring the one that wastes less shelf height.
            orientations = []
            a = p_len[pi]
            b = p_wid[pi]
            if a <= L + 1e-9 and b <= W + 1e-9:
                orientations.append((a, b, 0))
            if b <= L + 1e-9 and a <= W + 1e-9:
                orientations.append((b, a, 1))
            orientations.sort(key=lambda t: (-t[1], -t[0]))

            best_choice = None
            # Existing shelves first.
            for bi, stock_bin in enumerate(bins):
                shelves = stock_bin["shelves"]
                for sh_i, sh in enumerate(shelves):
                    sy, sh_h, sx = sh
                    for rw, rh, ori in orientations:
                        if rh <= sh_h + 1e-9 and sx + rw <= L + 1e-9:
                            waste_h = sh_h - rh
                            rem_x = L - (sx + rw)
                            score = waste_h * L + rem_x * 0.001
                            if best_choice is None or score < best_choice[0]:
                                best_choice = (score, bi, sh_i, rw, rh, ori)

            if best_choice is not None:
                _, bi, sh_i, rw, rh, ori = best_choice
                sh = bins[bi]["shelves"][sh_i]
                x = sh[2]
                y = sh[0]
                bins[bi]["placements"].append({
                    "piece": pi + 1,
                    "x": x,
                    "y": y,
                    "orientation": ori
                })
                sh[2] += rw
                placed = True

            if placed:
                continue

            # Try opening a new shelf in an existing stock.
            best_choice = None
            for bi, stock_bin in enumerate(bins):
                cur_y = stock_bin["height_used"]
                for rw, rh, ori in orientations:
                    if cur_y + rh <= W + 1e-9 and rw <= L + 1e-9:
                        score = (W - (cur_y + rh)) * L
                        if best_choice is None or score < best_choice[0]:
                            best_choice = (score, bi, rw, rh, ori)

            if best_choice is not None:
                _, bi, rw, rh, ori = best_choice
                y = bins[bi]["height_used"]
                bins[bi]["shelves"].append([y, rh, rw])
                bins[bi]["height_used"] += rh
                bins[bi]["placements"].append({
                    "piece": pi + 1,
                    "x": 0.0,
                    "y": y,
                    "orientation": ori
                })
                continue

            # Open a new stock instance.
            rw, rh, ori = orientations[0]
            if rw > L + 1e-9 or rh > W + 1e-9:
                return None
            bins.append({
                "height_used": rh,
                "shelves": [[0.0, rh, rw]],
                "placements": [{
                    "piece": pi + 1,
                    "x": 0.0,
                    "y": 0.0,
                    "orientation": ori
                }]
            })

        return [b["placements"] for b in bins]

    def build_solution(selected_stock_indices, counts):
        if time.time() > deadline:
            return None

        selected_stock_indices = list(selected_stock_indices)
        if len(selected_stock_indices) > 2:
            return None

        groups = {s: [] for s in selected_stock_indices}

        # Assign each piece type to the smallest selected stock that can contain it.
        for pi in range(m):
            if counts[pi] <= 0:
                continue
            candidates = [s for s in selected_stock_indices if fits_piece_stock(pi, s)]
            if not candidates:
                return None
            candidates.sort(key=lambda s: stock_area[s])
            chosen = candidates[0]
            groups[chosen].extend([pi] * counts[pi])

        placements = {}
        inst_id = 1
        total_stock = 0.0
        total_used = 0.0

        for si in selected_stock_indices:
            bins = pack_group(si, groups[si])
            if bins is None:
                return None
            for b in bins:
                if not b:
                    continue
                placements[inst_id] = {
                    "stock_type": si + 1,
                    "placements": b
                }
                inst_id += 1
                total_stock += stock_area[si]
                for pl in b:
                    total_used += p_area[pl["piece"] - 1]

        if total_stock <= 0:
            return {"objective": 0.0, "placements": {}}

        obj = max(0.0, (total_stock - total_used) / total_stock)
        return {"objective": obj, "placements": placements}

    # Candidate stock selections: singles and promising pairs.
    selections = []
    for s in range(n_stocks):
        selections.append((s,))

    for a in range(n_stocks):
        for b in range(a + 1, n_stocks):
            selections.append((a, b))

    # Prefer selections that can cover mandatory pieces and have small stock area.
    filtered = []
    for sel in selections:
        ok = True
        for pi in range(m):
            if p_min[pi] > 0 and not any(fits_piece_stock(pi, s) for s in sel):
                ok = False
                break
        if ok:
            filtered.append(sel)
    selections = filtered
    selections.sort(key=lambda sel: (len(sel), sum(stock_area[s] for s in sel)))

    # Generate count vectors.
    candidates = []
    base = list(p_min)
    candidates.append(base)

    total_min = sum(p_min)
    total_max = sum(p_max)

    # Add moderate optional pieces greedily by area, avoiding huge outputs.
    optional_order = sorted(range(m), key=lambda i: -p_area[i])

    for target_extra in (5, 10, 25, 50, 100, 200, 500):
        if total_min + target_extra > 2500:
            continue
        c = list(p_min)
        rem = target_extra
        changed = False
        for pi in optional_order:
            if rem <= 0:
                break
            add = min(p_max[pi] - c[pi], rem)
            if add > 0:
                c[pi] += add
                rem -= add
                changed = True
        if changed:
            candidates.append(c)

    if total_max <= 2000 and total_max > total_min:
        candidates.append(list(p_max))

    # Fractional max candidates if not too large.
    for frac in (0.25, 0.5, 0.75):
        c = []
        total = 0
        for i in range(m):
            val = p_min[i] + int((p_max[i] - p_min[i]) * frac)
            c.append(val)
            total += val
        if total <= 2500 and c not in candidates:
            candidates.append(c)

    best = None

    for counts in candidates:
        if time.time() > deadline:
            break
        for sel in selections:
            if time.time() > deadline:
                break
            sol = build_solution(sel, counts)
            if sol is None:
                continue
            if best is None or sol["objective"] < best["objective"] - 1e-12:
                best = sol

    # If no heuristic packing succeeded, construct a guaranteed valid one-piece-per-stock fallback.
    if best is None:
        placements = {}
        inst = 1
        total_stock = 0.0
        total_used = 0.0
        used_stock_types = []

        for pi in range(m):
            for _ in range(p_min[pi]):
                candidates = [s for s in range(n_stocks) if fits_piece_stock(pi, s)]
                if not candidates:
                    continue
                candidates.sort(key=lambda s: stock_area[s])

                # Respect at most two stock types.
                chosen = None
                for s in candidates:
                    if s in used_stock_types or len(used_stock_types) < 2:
                        chosen = s
                        break
                if chosen is None:
                    chosen = used_stock_types[0]

                if chosen not in used_stock_types:
                    used_stock_types.append(chosen)

                od = orient_dims(pi, chosen)
                if od is None:
                    continue
                _, _, ori = od
                placements[inst] = {
                    "stock_type": chosen + 1,
                    "placements": [{
                        "piece": pi + 1,
                        "x": 0.0,
                        "y": 0.0,
                        "orientation": ori
                    }]
                }
                inst += 1
                total_stock += stock_area[chosen]
                total_used += p_area[pi]

        obj = 1.0 if total_stock <= 0 else max(0.0, (total_stock - total_used) / total_stock)
        best = {"objective": obj, "placements": placements}

    return best