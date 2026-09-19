def dynamic_list_scheduling(times, machines, n_jobs, n_machines):
    """Dynamic list scheduling heuristic for job shop using LPT priority and event-driven machine availability.

    Maintains a ready set of operations whose predecessors are complete. Always selects the longest ready operation
    and assigns it to the machine that can start it earliest (accounting for current availability). Updates machine
    availability immediately and adds any successor operations to the ready set.

    Args:
        times: list[list[int]] — n_jobs x n_machines processing times
        machines: list[list[int]] — n_jobs x n_machines machine assignments (1-indexed)
        n_jobs: int
        n_machines: int

    Returns:
        start_times: list[list[int]] — n_jobs x n_machines start times (0-indexed internally)
    """
    import heapq

    # Operations: (priority=-p, job, op, machine, proc_time)
    ready = []
    for j in range(n_jobs):
        m = machines[j][0] - 1
        t = times[j][0]
        heapq.heappush(ready, (-t, j, 0, m, t))

    start_times = [[0 for _ in range(n_machines)] for _ in range(n_jobs)]
    machine_avail = [0] * n_machines

    while ready:
        _, j, i, m, t = heapq.heappop(ready)
        start = max(machine_avail[m], 0)
        start_times[j][i] = start
        machine_avail[m] = start + t
        if i < n_machines - 1:
            next_i = i + 1
            next_m = machines[j][next_i] - 1
            next_t = times[j][next_i]
            heapq.heappush(ready, (-next_t, j, next_i, next_m, next_t))

    return start_times