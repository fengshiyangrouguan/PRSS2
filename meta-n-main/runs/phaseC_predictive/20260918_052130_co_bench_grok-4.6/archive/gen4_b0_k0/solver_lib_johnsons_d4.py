def johnsons(jobs: list, matrix: list) -> list:
    """Apply Johnson's rule for F2||Cmax (optimal when m==2).
    Args:
        jobs: list[int] — 0-based job indices (size n).
        matrix: list[list[int]] — n rows, each [a_j, b_j] processing times on the two machines.
    Returns:
        list[int] — sequence of 0-based job indices in optimal order for Cmax.
    """
    if len(jobs) <= 1:
        return jobs
    a = [matrix[j][0] for j in jobs]
    b = [matrix[j][1] for j in jobs]
    group1 = [j for j in jobs if a[j] < b[j]]
    group2 = [j for j in jobs if a[j] > b[j]]
    group1.sort(key=lambda j: a[j])
    group2.sort(key=lambda j: b[j], reverse=True)
    return group1 + group2