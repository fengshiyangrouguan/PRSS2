def improved_batch_assignment(n_jobs: int, n_machines: int, permutation: list[int], processing_times: list[float]) -> list[int]:
    """Greedy list scheduling: assign jobs to machine with current min load.

    Args:
        n_jobs: int — number of jobs
        n_machines: int — number of machines
        permutation: list[int] — 1-based job order (from improved_permutation)
        processing_times: list[float] — processing time per job

    Returns:
        list[int] — batch_assignment (1-based machine id for each job in order)
    """
    loads = [0.0] * n_machines
    batch_assignment = []
    for job_id in permutation:
        machine = loads.index(min(loads))
        batch_assignment.append(machine + 1)
        loads[machine] += processing_times[job_id - 1]
    return batch_assignment