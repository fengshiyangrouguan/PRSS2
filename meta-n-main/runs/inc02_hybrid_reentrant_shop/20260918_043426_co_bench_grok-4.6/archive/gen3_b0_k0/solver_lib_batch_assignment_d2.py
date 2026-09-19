def batch_assignment(n_jobs: int, n_machines: int, processing_times: list) -> list:
    """Return batch_assignment that balances cumulative machine loads.

    Args:
        n_jobs: number of jobs
        n_machines: number of machines
        processing_times: list of total processing times per job

    Returns:
        list of machine ids (1-based) for each job
    """
    machine_loads = [0.0] * n_machines
    assignment = [0] * n_jobs
    for i in range(n_jobs):
        machine = min(range(n_machines), key=lambda m: machine_loads[m])
        assignment[i] = machine + 1
        machine_loads[machine] += processing_times[i]
    return assignment