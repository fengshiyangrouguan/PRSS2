def solve(**kwargs):
    n_jobs = kwargs['n_jobs']
    n_machines = kwargs['n_machines']
    # Simple heuristic: natural order for permutation (setup order)
    permutation = list(range(1, n_jobs + 1))
    # Round-robin batch assignment to machines for main processing and init
    batch_assignment = [(i % n_machines) + 1 for i in range(n_jobs)]
    return {
        'permutation': permutation,
        'batch_assignment': batch_assignment
    }