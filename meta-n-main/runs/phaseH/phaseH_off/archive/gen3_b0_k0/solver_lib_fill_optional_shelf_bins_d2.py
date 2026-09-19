def fill_optional_shelf_bins(
    bins: list,
    pieces: list,
    counts: list,
    stock_length: float,
    stock_width: float,
    deadline: float
) -> tuple:
    """Greedily add optional assortment pieces into existing shelf bins.

    Args:
        bins: list[dict] — existing shelf bins.  Each bin must have keys
            'shelves' as list of {'y': float, 'h': float, 'x': float},
            'placements' as list of placement dicts, and 'used_area' as float.
            Placements use 1-based piece ids and orientation 0/1.
        pieces: list[dict] — piece definitions with 'length', 'width', and
            optional 'max' fields.  The current count of each piece is given by
            counts.
        counts: list[int] — mutable current quantity placed for each piece.
            The function will increment entries but never exceed pieces[i]['max'].
        stock_length: float — usable stock sheet length / x dimension.
        stock_width: float — usable stock sheet width / y dimension.
        deadline: float — absolute time.time() deadline.  The function stops
            early if this deadline is reached.

    Returns:
        tuple[list, list] — the mutated bins and counts after greedily inserting
        optional pieces that fit into already-open shelf space.
    """
    import time

    eps = 1e-9
    n = len(pieces)

    def opts(i: int) -> list:
        pl = float(pieces[i].get("length", 0.0))
        pw = float(pieces[i].get("width", 0.0))
        out = []
        if pl <= stock_length + eps and pw <= stock_width + eps:
            out.append((pl, pw, 0))
        if abs(pl - pw) > eps and pw <= stock_length + eps and pl <= stock_width + eps:
            out.append((pw, pl, 1))
        return out

    area = [
        float(pieces[i].get("length", 0.0)) * float(pieces[i].get("width", 0.0))
        for i in range(n)
    ]

    # Large pieces first; this usually improves utilization of shelf packings.
    candidates = sorted(range(n), key=lambda i: area[i], reverse=True)

    improved = True
    while improved and time.time() < deadline:
        improved = False
        best = None

        for i in candidates:
            if time.time() >= deadline:
                break

            mx = int(pieces[i].get("max", counts[i] if i < len(counts) else 0))
            if counts[i] >= mx:
                continue

            for rw, rh, ori in opts(i):
                # Try appending to an existing shelf.
                for bi, b in enumerate(bins):
                    for si, sh in enumerate(b.get("shelves", [])):
                        if rh <= float(sh["h"]) + eps and float(sh["x"]) + rw <= stock_length + eps:
                            rem_x = stock_length - (float(sh["x"]) + rw)
                            height_waste = float(sh["h"]) - rh
                            # Prefer high area, then tight shelf fit.
                            score = (-area[i], height_waste, rem_x, bi, si)
                            if best is None or score < best[0]:
                                best = (score, i, bi, si, rw, rh, ori, "shelf")

                    # Try opening a new shelf inside this existing bin.
                    next_y = float(b.get("next_y", 0.0))
                    if next_y + rh <= stock_width + eps:
                        rem_y = stock_width - (next_y + rh)
                        rem_x = stock_length - rw
                        score = (-area[i], rem_y, rem_x, bi, -1)
                        if best is None or score < best[0]:
                            best = (score, i, bi, -1, rw, rh, ori, "new_shelf")

        if best is None:
            break

        _, i, bi, si, rw, rh, ori, mode = best
        b = bins[bi]

        if mode == "shelf":
            sh = b["shelves"][si]
            x = float(sh["x"])
            y = float(sh["y"])
            sh["x"] = x + rw
        else:
            x = 0.0
            y = float(b.get("next_y", 0.0))
            b.setdefault("shelves", []).append({"y": y, "h": rh, "x": rw})
            b["next_y"] = y + rh

        b.setdefault("placements", []).append({
            "piece": i + 1,
            "x": x,
            "y": y,
            "orientation": ori
        })
        b["used_area"] = float(b.get("used_area", 0.0)) + rw * rh
        counts[i] += 1
        improved = True

    return bins, counts