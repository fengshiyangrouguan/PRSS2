def solve(**kwargs):
    n_jobs = kwargs['n_jobs']
    n_machines = kwargs['n_machines']
    # Simple valid heuristic: natural permutation and round-robin batch assignment
    permutation = list(range(1, n_jobs + 1))
    batch_assignment = [(i % n_machines) + 1 for i in range(n_jobs)]
    return {
        'permutation': permutation,
        'batch_assignment': batch_assignment
    }