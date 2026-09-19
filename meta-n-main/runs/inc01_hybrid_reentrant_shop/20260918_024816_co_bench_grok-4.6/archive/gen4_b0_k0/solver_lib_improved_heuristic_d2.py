def improved_heuristic(n_jobs: int, n_machines: int) -> dict:
    """Improved list-scheduling heuristic for hybrid reentrant shop scheduling.

    Args:
        n_jobs: int — number of jobs to schedule
        n_machines: int — number of machines in the shop

    Returns:
        dict — must contain exactly:
            'permutation': list[int] — job order (1-based indices)
            'batch_assignment': list[int] — machine id (1-based) for each job
    """
    import random
    # Maintain a reasonable permutation order (could be randomized or sorted if metadata available)
    permutation = list(range(1, n_jobs + 1))
    # Greedy assignment: always assign next job to the currently least-loaded machine
    # This balances load and reduces makespan impact for reentrant visits
    batch_assignment = []
    loads = [0] * n_machines
    for _ in range(n_jobs):
        machine = loads.index(min(loads))
        batch_assignment.append(machine + 1)
        loads[machine] += 1
    return {
        'permutation': permutation,
        'batch_assignment': batch_assignment
    }