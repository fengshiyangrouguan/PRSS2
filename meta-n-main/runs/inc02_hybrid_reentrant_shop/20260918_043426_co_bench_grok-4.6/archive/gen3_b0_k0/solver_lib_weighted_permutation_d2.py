def weighted_permutation(jobs: list, setup_times: list, processing_times: list) -> list:
    """Return job permutation sorted by weighted key (setup dominant, processing secondary).

    Args:
        jobs: list of job indices (0..n_jobs-1)
        setup_times: list of setup times for each job
        processing_times: list of total processing times for each job

    Returns:
        list of job indices in optimal order for the solver
    """
    def key(i):
        return setup_times[i] * 10 + processing_times[i] / 100
    return sorted(range(len(jobs)), key=key, reverse=True)