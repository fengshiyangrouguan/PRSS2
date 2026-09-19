def dynamic_lpt(times, machines, n_jobs, n_machines):
    """Dynamic list scheduling for job-shop: LPT priority + precedence + dynamic dispatch.

    Maintains a ready queue of operations whose predecessors are finished.
    Always schedules the longest-remaining-operation ready job on the machine that
    can start it earliest.

    Args:
        times: list[list[int]] — n_jobs x n_machines processing times
        machines: list[list[int]] — n_jobs x n_machines machine assignments (1-indexed)
        n_jobs: int
        n_machines: int

    Returns:
        start_times: list[list[int]] — n_jobs x n_machines start times (0-indexed)
    """
    import heapq

    machine_avail = [0] * n_machines
    start_times = [[0 for _ in range(n_machines)] for _ in range(n_jobs)]
    scheduled = [[False] * n_machines for _ in range(n_jobs)]
    ready = []

    # Initial push: first ops of every job
    for j in range(n_jobs):
        if n_machines > 0:
            m = machines[j][0] - 1
            p = times[j][0]
            heapq.heappush(ready, (-p, j, 0, m))  # priority, job, op, machine

    while ready:
        _, j, i, m = heapq.heappop(ready)
        if scheduled[j][i]:
            continue

        # Precedence check
        if i > 0:
            if not scheduled[j][i - 1]:
                heapq.heappush(ready, (-times[j][i], j, i, m))
                continue

        # Compute start
        pred_finish = start_times[j][i - 1] + times[j][i - 1] if i > 0 else 0
        start = max(pred_finish, machine_avail[m])

        start_times[j][i] = start
        machine_avail[m] = start + times[j][i]
        scheduled[j][i] = True

        # Release successor
        if i < n_machines - 1:
            next_m = machines[j][i + 1] - 1
            next_p = times[j][i + 1]
            heapq.heappush(ready, (-next_p, j, i + 1, next_m))

    return start_times