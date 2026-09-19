def local_search(initial_seq: list, matrix: list, time_limit: float = 5.0) -> list:
    """Improve an initial job sequence via limited random adjacent swaps.

    Args:
        initial_seq: list of 0-based job indices (the NEH output).
        matrix: 2-D list of processing times (same as in solve()).
        time_limit: maximum seconds to run the search.

    Returns:
        list — the improved job sequence (0-based indices).
    """
    import time, random
    if len(initial_seq) < 3:
        return initial_seq[:]
    best_seq = initial_seq[:]
    best_m = makespan(best_seq, matrix)
    start = time.time()
    while time.time() - start < time_limit:
        i = random.randint(0, len(best_seq) - 2)
        neighbor = best_seq[:]
        neighbor[i], neighbor[i + 1] = neighbor[i + 1], neighbor[i]
        cm = makespan(neighbor, matrix)
        if cm < best_m:
            best_seq = neighbor
            best_m = cm
    return best_seq