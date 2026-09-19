def compute_makespan(seq: list[int], matrix: list[list[int]], m: int) -> int:
    """Compute makespan for a given job sequence using incremental machine times.

    Args:
        seq: list[int] — 0-based job indices in schedule order
        matrix: list[list[int]] — processing time matrix; matrix[j][k] is job j on machine k
        m: int — number of machines (columns)

    Returns:
        int — makespan (maximum machine completion time)
    """
    times = [0] * m
    for job in seq:
        for k in range(m):
            times[k] += matrix[job][k]
            if k > 0:
                times[k] = max(times[k], times[k - 1])
    return times[-1]