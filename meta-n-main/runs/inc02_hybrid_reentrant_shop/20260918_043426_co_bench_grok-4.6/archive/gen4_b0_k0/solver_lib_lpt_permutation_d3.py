def lpt_permutation(n_jobs: int, processing_times: list[float]) -> list[int]:
    """Return 1-based job permutation sorted by longest processing time (descending).

    Args:
        n_jobs: int — number of jobs in the instance
        processing_times: list[float] — processing time per job

    Returns:
        list[int] — permutation of job numbers (1-based) sorted by processing time descending
    """
    return [p + 1 for p in sorted(range(n_jobs), key=lambda i: processing_times[i], reverse=True)]