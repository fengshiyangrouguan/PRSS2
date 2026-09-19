def solve(**kwargs):
    import time
    import math
    import itertools

    start_time = time.time()
    deadline = start_time + 8.5

    stocks = kwargs.get("stocks", [])
    pieces = kwargs.get("pieces", [])
    m = kwargs.get("m", len(pieces))

    if not stocks:
        return {"objective": 0.0, "placements": {}}

    nstocks = len(stocks)

    stock_dims = []
    stock_area = []
    for s in stocks:
        L = float(s.get("length", 0.0))
        W = float(s.get("width", 0.0))
        stock_dims.append((L, W))
        stock_area.append(L * W)

    piece_dims = []
    piece_area = []
    mins = []
    maxs = []
    for p in pieces:
        L = float(p.get("length", 0.0))
        W = float(p.get("width", 0.0))
        mn = int(p.get("min", 0))
        mx = int(p.get("max", mn))
        if mx < mn:
            mx = mn
        piece_dims.append((L, W))
        piece_area.append(L * W)
        mins.append(mn)
        maxs.append(mx)

    eps = 1e-9

    def fits_piece_stock(pi, si):
        pl, pw = piece_dims[pi]
        sl, sw = stock_dims[si]
        return (pl <= sl + eps and pw <= sw + eps) or (pw <= sl + eps and pl <= sw + eps)

    def rect_contains(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        return (
            bx >= ax - eps and by >= ay - eps and
            bx + bw <= ax + aw + eps and
            by + bh <= ay + ah + eps
        )

    def prune_free(free_rects):
        cleaned = []
        for r in free_rects:
            if r[2] <= eps or r[3] <= eps:
                continue
            duplicate = False
            for q in cleaned:
                if rect_contains(q, r):
                    duplicate = True
                    break
            if duplicate:
                continue
            cleaned = [q for q in cleaned if not rect_contains(r, q)]
            cleaned.append(r)
        return cleaned

    def try_place_in_instance(inst, pi):
        pl, pw = piece_dims[pi]
        best = None
        best_score = None

        for ri, fr in enumerate(inst["free"]):
            fx, fy, fw, fh = fr
            orientations = []
            if pl <= fw + eps and pw <= fh + eps:
                orientations.append((0, pl, pw))
            if pw <= fw + eps and pl <= fh + eps and abs(pl - pw) > eps:
                orientations.append((1, pw, pl))

            for orient, w, h in orientations:
                leftover_area = fw * fh - w * h
                short_side = min(fw - w, fh - h)
                long_side = max(fw - w, fh - h)
                score = (leftover_area, short_side, long_side)
                if best_score is None or score < best_score:
                    best_score = score
                    best = (ri, fx, fy, w, h, orient)

        if best is None:
            return False

        ri, x, y, w, h, orient = best
        fx, fy, fw, fh = inst["free"][ri]

        new_free = inst["free"][:ri] + inst["free"][ri + 1:]

        # Simple guillotine split: right rectangle and top rectangle.
        right_w = fw - w
        top_h = fh - h
        if right_w > eps:
            new_free.append((x + w, y, right_w, h))
        if top_h > eps:
            new_free.append((x, y + h, fw, top_h))

        inst["free"] = prune_free(new_free)
        inst["placements"].append({
            "piece": pi + 1,
            "x": x,
            "y": y,
            "orientation": orient
        })
        inst["used_area"] += piece_area[pi]
        return True

    def clone_solution(instances):
        out = []
        for inst in instances:
            out.append({
                "stock_type": inst["stock_type"],
                "free": list(inst["free"]),
                "placements": [dict(p) for p in inst["placements"]],
                "used_area": inst["used_area"]
            })
        return out

    def pack_for_subset(subset, mode):
        # mode controls optional aggressiveness.
        instances = []
        counts = [0] * m

        mandatory_items = []
        for i in range(m):
            for _ in range(mins[i]):
                mandatory_items.append(i)

        mandatory_items.sort(key=lambda i: (-piece_area[i], -max(piece_dims[i])))

        def open_new_and_place(pi):
            best_si = None
            best_area = None
            for si in subset:
                if fits_piece_stock(pi, si):
                    if best_area is None or stock_area[si] < best_area:
                        best_area = stock_area[si]
                        best_si = si
            if best_si is None:
                return False

            sl, sw = stock_dims[best_si]
            inst = {
                "stock_type": best_si,
                "free": [(0.0, 0.0, sl, sw)],
                "placements": [],
                "used_area": 0.0
            }
            ok = try_place_in_instance(inst, pi)
            if not ok:
                return False
            instances.append(inst)
            return True

        def place_existing(pi):
            best_idx = None
            best_snapshot = None
            best_score = None
            for idx, inst in enumerate(instances):
                tmp = {
                    "stock_type": inst["stock_type"],
                    "free": list(inst["free"]),
                    "placements": [dict(p) for p in inst["placements"]],
                    "used_area": inst["used_area"]
                }
                if try_place_in_instance(tmp, pi):
                    remaining_free = sum(r[2] * r[3] for r in tmp["free"])
                    score = (remaining_free, len(tmp["free"]))
                    if best_score is None or score < best_score:
                        best_score = score
                        best_idx = idx
                        best_snapshot = tmp
            if best_idx is None:
                return False
            instances[best_idx] = best_snapshot
            return True

        for pi in mandatory_items:
            if time.time() > deadline:
                return None
            if not place_existing(pi):
                if not open_new_and_place(pi):
                    return None
            counts[pi] += 1

        # Optional filling: primarily use existing free space; in aggressive mode,
        # also open a new stock if the piece itself fills it reasonably well.
        optional_order = []
        for i in range(m):
            extra = maxs[i] - counts[i]
            for _ in range(extra):
                optional_order.append(i)

        optional_order.sort(key=lambda i: (-piece_area[i], -max(piece_dims[i])))

        for pi in optional_order:
            if time.time() > deadline:
                break
            if counts[pi] >= maxs[pi]:
                continue
            if place_existing(pi):
                counts[pi] += 1
            elif mode == "aggressive":
                best_fill = 0.0
                best_si = None
                for si in subset:
                    if fits_piece_stock(pi, si) and stock_area[si] > eps:
                        fill = piece_area[pi] / stock_area[si]
                        if fill > best_fill:
                            best_fill = fill
                            best_si = si
                if best_si is not None and best_fill >= 0.72:
                    old_len = len(instances)
                    if open_new_and_place(pi):
                        counts[pi] += 1
                    else:
                        instances = instances[:old_len]

        for i in range(m):
            if counts[i] < mins[i] or counts[i] > maxs[i]:
                return None

        return instances

    subsets = []
    for i in range(nstocks):
        subsets.append((i,))
    for i in range(nstocks):
        for j in range(i + 1, nstocks):
            subsets.append((i, j))

    # Prefer smaller stock areas first for deterministic fallback.
    subsets.sort(key=lambda sub: (len(sub), sum(stock_area[i] for i in sub)))

    best_instances = None
    best_obj = None

    for subset in subsets:
        if time.time() > deadline:
            break

        possible = True
        for pi in range(m):
            if mins[pi] > 0 and not any(fits_piece_stock(pi, si) for si in subset):
                possible = False
                break
        if not possible:
            continue

        for mode in ("normal", "aggressive"):
            if time.time() > deadline:
                break
            insts = pack_for_subset(subset, mode)
            if insts is None:
                continue

            total_stock = sum(stock_area[inst["stock_type"]] for inst in insts)
            total_used = sum(inst["used_area"] for inst in insts)

            if total_stock <= eps:
                obj = 0.0
            else:
                obj = max(0.0, min(1.0, (total_stock - total_used) / total_stock))

            if best_obj is None or obj < best_obj - 1e-12:
                best_obj = obj
                best_instances = clone_solution(insts)

    # If no feasible solution was found, create the safest possible fallback:
    # each required piece alone in the smallest stock that can contain it.
    if best_instances is None:
        fallback = []
        feasible = True
        for pi in range(m):
            for _ in range(mins[pi]):
                best_si = None
                best_a = None
                for si in range(nstocks):
                    if fits_piece_stock(pi, si):
                        if best_a is None or stock_area[si] < best_a:
                            best_a = stock_area[si]
                            best_si = si
                if best_si is None:
                    feasible = False
                    break
                sl, sw = stock_dims[best_si]
                inst = {
                    "stock_type": best_si,
                    "free": [(0.0, 0.0, sl, sw)],
                    "placements": [],
                    "used_area": 0.0
                }
                try_place_in_instance(inst, pi)
                fallback.append(inst)
            if not feasible:
                break

        if feasible:
            # Enforce at most two stock types by keeping only the two most common;
            # if impossible, still return best-effort formatted output.
            used_types = {}
            for inst in fallback:
                used_types[inst["stock_type"]] = used_types.get(inst["stock_type"], 0) + 1
            if len(used_types) <= 2:
                best_instances = fallback
                total_stock = sum(stock_area[inst["stock_type"]] for inst in best_instances)
                total_used = sum(inst["used_area"] for inst in best_instances)
                best_obj = 0.0 if total_stock <= eps else max(0.0, min(1.0, (total_stock - total_used) / total_stock))

    if best_instances is None:
        # Last resort, valid format.
        return {
            "objective": 1.0,
            "placements": {
                1: {
                    "stock_type": 1,
                    "placements": []
                }
            }
        }

    placements_out = {}
    for idx, inst in enumerate(best_instances, 1):
        placements_out[idx] = {
            "stock_type": inst["stock_type"] + 1,
            "placements": inst["placements"]
        }

    if best_obj is None:
        total_stock = sum(stock_area[inst["stock_type"]] for inst in best_instances)
        total_used = sum(inst["used_area"] for inst in best_instances)
        best_obj = 0.0 if total_stock <= eps else max(0.0, min(1.0, (total_stock - total_used) / total_stock))

    return {
        "objective": float(best_obj),
        "placements": placements_out
    }