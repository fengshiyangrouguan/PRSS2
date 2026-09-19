def solve(**kwargs):
    """
    Problem:
        Given an instance of the Aircraft Landing Scheduling Problem, schedule the landing time for each plane and assign a runway so that:
          - Each landing time is within its allowed time window.
          - Each plane is assigned to one runway (from the available runways).
          - For any two planes assigned to the same runway, if plane i lands at or before plane j, then the landing times must be separated by at least
            the specified separation time (provided in the input data).
          - The overall penalty is minimized. For each plane, if its landing time is earlier than its target time, a penalty
            is incurred proportional to the earliness; if later than its target time, a penalty proportional to the lateness is incurred.
          - If any constraint is violated, the solution receives no score.

    Input kwargs:
        num_planes  : (int) Number of planes.
        num_runways : (int) Number of runways.
        freeze_time : (float) Freeze time (unused in scheduling decisions).
        planes      : (list of dict) Each dictionary contains:
                        - "appearance"    : float, time the plane appears.
                        - "earliest"      : float, earliest landing time.
                        - "target"        : float, target landing time.
                        - "latest"        : float, latest landing time.
                        - "penalty_early" : float, penalty per unit time landing early.
                        - "penalty_late"  : float, penalty per unit time landing late.
        separation  : (list of lists) separation[i][j] is the required gap after plane i lands before plane j can land
                      when they are assigned to the same runway.

    Returns:
        A dictionary named "schedule" mapping each plane id (1-indexed) to a dictionary with its scheduled landing time
        and assigned runway, e.g., { plane_id: {"landing_time": float, "runway": int}, ... }.
    """
    schedule = {}
    planes = kwargs["planes"]
    n = len(planes)
    # Sort planes by target time to establish a candidate landing order
    sorted_idx = sorted(range(n), key=lambda i: planes[i]["target"])
    prev_t = -float('inf')
    for rank, idx in enumerate(sorted_idx):
        plane = planes[idx]
        pid = idx + 1
        earliest = plane["earliest"]
        target = plane["target"]
        latest = plane["latest"]
        if rank == 0:
            t = target
        else:
            prev_idx = sorted_idx[rank - 1]
            sep = kwargs["separation"][prev_idx][idx]
            t = max(earliest, prev_t + sep)
        if t > latest:
            t = latest
        schedule[pid] = {"landing_time": t, "runway": 1}
        prev_t = t
    return {"schedule": schedule}