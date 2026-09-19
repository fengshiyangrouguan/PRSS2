def improved_permutation(n_jobs: int, setup_times: list[float], processing_times: list[float]) -> list[int]:
    """Return 1-based job permutation sorted by hybrid key (setup * processing descending).

    Args:
        n_jobs: int — number of jobs in the instance
        setup_times: list[float] — setup time per job (scalar per job)
        processing_times: list[float] — processing time per job

    Returns:
        list[int] — permutation of job numbers (1-based) sorted by setup*processing descending
    """
    def key(i):
        return setup_times[i] * processing_times[i]
    return [p + 1 for p in sorted(range(n_jobs), key=key, reverse=True)]