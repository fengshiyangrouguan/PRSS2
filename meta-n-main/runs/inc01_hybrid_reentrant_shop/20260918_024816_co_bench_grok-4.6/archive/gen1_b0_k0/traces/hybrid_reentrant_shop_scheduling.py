def solve(**kwargs):
    n_jobs = kwargs['n_jobs']
    n_machines = kwargs['n_machines']
    init_time = kwargs['init_time']
    setup_times = kwargs['setup_times']
    processing_times = kwargs['processing_times']

    # Sort jobs by descending total time (init + setup + main) for longest-processing-time-first on server
    totals = [init_time + setup_times[i] + processing_times[i] for i in range(n_jobs)]
    sorted_jobs = sorted(range(n_jobs), key=lambda i: totals[i], reverse=True)
    permutation = [j + 1 for j in sorted_jobs]

    # Round-robin batch assignment to primary machines (1-based)
    batch_assignment = [(i % n_machines) + 1 for i in range(n_jobs)]

    return {
        'permutation': permutation,
        'batch_assignment': batch_assignment
    }