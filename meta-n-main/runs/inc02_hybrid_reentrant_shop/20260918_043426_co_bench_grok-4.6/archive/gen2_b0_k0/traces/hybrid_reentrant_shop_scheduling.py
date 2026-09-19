def solve(**kwargs):
    """
    Input:
      - n_jobs: Integer; the number of jobs.
      - n_machines: Integer; the number of primary machines.
      - init_time: Integer; the initialization time for every job on a primary machine.
      - setup_times: List of integers; the setup times for each job on the remote server.
      - processing_times: List of integers; the processing times for each job in the main processing stage.

    Output:
      A dictionary with the following keys:
        - 'permutation': A list of integers of length n_jobs. This list represents the order in which the jobs are processed on the remote server.
        - 'batch_assignment': A list of integers of length n_jobs. Each element indicates the primary machine to which the corresponding job (or batch) is assigned.
    """
    n_jobs = kwargs['n_jobs']
    n_machines = kwargs['n_machines']
    setup_times = kwargs['setup_times']
    processing_times = kwargs['processing_times']

    # Improved permutation: sort jobs by hybrid key setup_times[i] * processing_times[i] descending
    # to prioritize jobs with both high setup and high processing
    permutation = sorted(range(n_jobs), key=lambda j: setup_times[j] * processing_times[j], reverse=True)
    permutation = [p + 1 for p in permutation]  # 1-based as per convention

    # Improved batch_assignment: list scheduling on hybrid_permutation order
    # assign each job to the machine with the current minimum load using processing_times
    loads = [0] * n_machines
    batch_assignment = []
    for j in permutation:
        min_load = min(loads)
        min_idx = loads.index(min_load)
        batch_assignment.append(min_idx + 1)
        loads[min_idx] += processing_times[j]

    return {
        'permutation': permutation,
        'batch_assignment': batch_assignment
    }