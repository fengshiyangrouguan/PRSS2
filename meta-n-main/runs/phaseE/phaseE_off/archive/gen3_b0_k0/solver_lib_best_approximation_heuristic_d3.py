def best_approximation_heuristic(n: int, m: int, matrix: list) -> list:
    """Return job sequence (0-based) via total processing time sorting heuristic.

    Args:
        n: int — number of jobs.
        m: int — number of machines.
        matrix: list[list[float]] — processing times, shape (n, m).

    Returns:
        list[int] — 0-based job indices in processing order.
    """
    if n == 0:
        return []
    totals = [sum(row) for row in matrix]
    order = sorted(range(n), key=lambda i: totals[i], reverse=True)
    return order