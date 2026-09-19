def reentrant_list_scheduling(n_jobs: int, n_machines: int) -> dict:
    """Reentrant-aware list scheduling for hybrid reentrant shop scheduling.

    Args:
        n_jobs: int — number of jobs
        n_machines: int — number of machines

    Returns:
        dict — must contain exactly:
            'permutation': list[int] — 1-based job order prioritizing longer/reentrant-critical jobs first
            'batch_assignment': list[int] — machine id (1-based) for each job in permutation order
    """
    import random
    # Priority permutation: sort jobs by index (higher = more reentrant-critical assumed)
    permutation = sorted(range(1, n_jobs + 1), key=lambda x: -x)
    # Reentrant look-ahead greedy: assign each job to current least-loaded machine while simulating 2-3 typical revisits
    batch_assignment = []
    loads = [0] * n_machines
    for job in permutation:
        # Simulate reentrant load: increment for initial + 2 revisits (common in hybrid reentrant model)
        for _ in range(3):
            machine = loads.index(min(loads))
            loads[machine] += 1
        batch_assignment.append(machine + 1)
    return {
        'permutation': permutation,
        'batch_assignment': batch_assignment
    }