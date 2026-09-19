def solve(**kwargs):
    n_jobs = kwargs['n_jobs']
    n_machines = kwargs['n_machines']
    init_time = kwargs['init_time']
    setup_times = kwargs['setup_times']
    processing_times = kwargs['processing_times']
    batch_assignment = [(i % n_machines) + 1 for i in range(n_jobs)]
    permutation = [p + 1 for p in sorted(range(n_jobs), key=lambda i: setup_times[i], reverse=True)]
    return {
        'permutation': permutation,
        'batch_assignment': batch_assignment
    }