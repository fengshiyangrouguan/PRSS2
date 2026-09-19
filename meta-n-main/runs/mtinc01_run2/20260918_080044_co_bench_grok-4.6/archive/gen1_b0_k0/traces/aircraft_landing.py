def solve(**kwargs):
    n = kwargs["num_planes"]
    r = kwargs["num_runways"]
    planes = kwargs["planes"]
    sep = kwargs["separation"]
    schedule = {}
    order = sorted(range(n), key=lambda i: planes[i]["target"])
    last_time = [float("-inf")] * r
    last_plane_idx = [-1] * r
    for ord_idx in range(n):
        idx = order[ord_idx]
        plane_id = idx + 1
        target = planes[idx]["target"]
        earliest = planes[idx]["earliest"]
        appearance = planes[idx]["appearance"]
        latest = planes[idx]["latest"]
        lower = max(appearance, earliest)
        best_lb = float("inf")
        chosen_r = -1
        for runway in range(r):
            prev_idx = last_plane_idx[runway]
            if prev_idx == -1:
                prev_t = float("-inf")
                prev_sep_idx = -1
            else:
                prev_t = planes[prev_idx]["target"]
                prev_sep_idx = prev_idx
            lb = max(lower, prev_t + sep[prev_sep_idx][idx] if prev_sep_idx != -1 else lower)
            if lb <= latest and lb < best_lb:
                best_lb = lb
                chosen_r = runway
        if chosen_r == -1:
            runway = 0
            prev_idx = last_plane_idx[0]
            if prev_idx == -1:
                prev_t = float("-inf")
                prev_sep_idx = -1
            else:
                prev_t = planes[prev_idx]["target"]
                prev_sep_idx = prev_idx
            best_lb = max(lower, prev_t + sep[prev_sep_idx][idx] if prev_sep_idx != -1 else lower)
            chosen_r = 0
        last_plane_idx[chosen_r] = idx
        landing_time = max(target, best_lb)
        last_time[chosen_r] = landing_time
        schedule[plane_id] = {"landing_time": landing_time, "runway": chosen_r + 1}
    return {"schedule": schedule}