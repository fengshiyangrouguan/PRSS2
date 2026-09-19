from collections import defaultdict

def solve(**kwargs):
    num_planes = kwargs["num_planes"]
    num_runways = kwargs["num_runways"]
    planes_list = kwargs["planes"]
    sep = kwargs["separation"]
    if num_planes == 0:
        return {"schedule": {}}
    # Clamp targets to windows
    for p in planes_list:
        p["window_low"] = max(p["appearance"], p["earliest"])
        p["window_high"] = p["latest"]
    # Sort planes by target time
    sorted_planes = sorted(range(num_planes), key=lambda i: planes_list[i]["target"])
    # Assign runways round-robin in target order
    runway_assignment = [0] * num_planes
    for i, pid in enumerate(sorted_planes):
        r = i % num_runways
        runway_assignment[pid] = r
    runway_planes = defaultdict(list)
    for pid in range(num_planes):
        runway_planes[runway_assignment[pid]].append(pid)
    schedule = {}
    for r, plane_ids in runway_planes.items():
        if not plane_ids:
            continue
        sorted_ids = sorted(plane_ids, key=lambda pid: planes_list[pid]["target"])
        k = len(sorted_ids)
        t = [0.0] * k
        for j in range(k):
            pid = sorted_ids[j]
            t[j] = max(planes_list[pid]["window_low"], min(planes_list[pid]["target"], planes_list[pid]["window_high"]))
        # Enforce consecutive separations (in target order)
        for j in range(1, k):
            pid_prev = sorted_ids[j-1]
            pid = sorted_ids[j]
            sep_val = sep[pid_prev][pid]
            t[j] = max(t[j], t[j-1] + sep_val)
        # Enforce all pairwise separations by propagating requirements
        for _ in range(k):  # multiple passes to propagate
            for j in range(k):
                for m in range(j + 1, k):
                    pid1 = sorted_ids[j]
                    pid2 = sorted_ids[m]
                    required = t[j] + sep[pid1][pid2]
                    if t[m] < required:
                        t[m] = required
        # Clamp to latest
        for j in range(k):
            pid = sorted_ids[j]
            t[j] = min(t[j], planes_list[pid]["window_high"])
        # Assign to schedule
        for j, pid in enumerate(sorted_ids):
            schedule[pid + 1] = {"landing_time": t[j], "runway": r}
    return {"schedule": schedule}