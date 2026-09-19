def solve(**kwargs):
    """
    Hybrid Reentrant Shop Scheduling heuristic.
    permutation: jobs sorted descending by setup_times[i] * processing_times[i] (hybrid priority for high-setup/high-proc jobs to reduce makespan).
    batch_assignment: round-robin assignment to n_machines (1-based).
    """
    n_jobs = kwargs['n_jobs']
    n_machines = kwargs['n_machines']
    setup_times = kwargs['setup_times']
    processing_times = kwargs['processing_times']
    
    # Hybrid permutation: sort by product of setup and processing times (descending)
    jobs = list(range(n_jobs))
    jobs.sort(key=lambda i: setup_times[i] * processing_times[i], reverse=True)
    permutation = [j + 1 for j in jobs]  # 1-based
    
    # Round-robin batch assignment to primary machines
    batch_assignment = [(i % n_machines) + 1 for i in range(n_jobs)]
    
    return {
        'permutation': permutation,
        'batch_assignment': batch_assignment
    }