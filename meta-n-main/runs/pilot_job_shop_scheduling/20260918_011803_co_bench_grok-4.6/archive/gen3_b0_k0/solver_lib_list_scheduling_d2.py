def list_scheduling(times, machines, n_jobs, n_machines):
    """List scheduling heuristic for job shop scheduling using LPT priority.

    Args:
        times: list[list[int]] — n_jobs x n_machines processing times
        machines: list[list[int]] — n_jobs x n_machines machine assignments (1-indexed)
        n_jobs: int
        n_machines: int

    Returns:
        start_times: list[list[int]] — n_jobs x n_machines start times (0-indexed internally)
    """
    import heapq

    # Flatten all operations with LPT key
    operations = []
    for j in range(n_jobs):
        for i in range(n_machines):
            m = machines[j][i] - 1
            t = times[j][i]
            # priority = processing time (longest first)
            heapq.heappush(operations, (-t, j, i, m, t))

    start_times = [[0 for _ in range(n_machines)] for _ in range(n_jobs)]
    machine_avail = [0] * n_machines

    while operations:
        _, j, i, m, t = heapq.heappop(operations)
        start = machine_avail[m]
        start_times[j][i] = start
        machine_avail[m] = start + t

    return start_times