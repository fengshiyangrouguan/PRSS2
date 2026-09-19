def least_loaded_assignment(n_jobs, n_machines):
    """Assign jobs to machines using Least Loaded Machine heuristic for better load balancing.

    Args:
        n_jobs: int — number of jobs to schedule.
        n_machines: int — number of available machines.

    Returns:
        list[int] — length n_jobs list of machine IDs (1-based) for each job's batch.
    """
    import heapq
    # Min-heap of (current_load, machine_id)
    load_heap = [(0, i + 1) for i in range(n_machines)]
    assignment = []
    for j in range(1, n_jobs + 1):
        load, machine = heapq.heappop(load_heap)
        assignment.append(machine)
        heapq.heappush(load_heap, (load + 1, machine))
    return assignment