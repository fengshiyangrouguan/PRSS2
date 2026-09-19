import solver_lib

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

    # LPT permutation (1-based)
    permutation = solver_lib.lpt_permutation(n_jobs, processing_times)
    # Improve with hybrid key (setup * processing descending)
    permutation = solver_lib.improved_permutation(n_jobs, setup_times, processing_times, permutation)

    # Round-robin batch assignment
    batch_assignment = [(i % n_machines) + 1 for i in range(n_jobs)]

    return {
        'permutation': permutation,
        'batch_assignment': batch_assignment
    }