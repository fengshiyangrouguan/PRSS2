def johnson_heuristic(n: int, m: int, matrix: list) -> list:
    """Johnson's heuristic for m-machine flow shop (approximation algorithm).

    Args:
        n: number of jobs
        m: number of machines
        matrix: n x m list of processing times (matrix[i][j] = time of job i on machine j)

    Returns:
        list of job indices (0-based) in the order to process them.
    """
    import math
    # For m==2 use optimal Johnson's rule
    if m == 2:
        times = [(matrix[i][0] + matrix[i][1], matrix[i][1] - matrix[i][0]) for i in range(n)]
        jobs = list(range(n))
        jobs.sort(key=lambda i: (times[i][0], -times[i][1]) if times[i][1] < 0 else (times[i][0], math.inf))
        return jobs
    # For m>2 use NEH heuristic (insertion heuristic, fast and strong)
    else:
        # Start with longest processing time order
        p = [sum(row) for row in matrix]
        jobs = list(range(n))
        jobs.sort(key=p.__getitem__, reverse=True)
        # NEH insertion
        for i in range(1, n):
            best_pos = i
            best_makespan = float('inf')
            for pos in range(i + 1):
                # insert job jobs[i] at position pos
                seq = jobs[:pos] + [jobs[i]] + jobs[pos:i]
                # compute makespan of this seq (simple O(n*m) simulation)
                comp = [0.0] * m
                for job_idx in seq:
                    for mm in range(m):
                        comp[mm] = max(comp[mm], comp[mm - 1] if mm > 0 else 0) + matrix[job_idx][mm]
                if comp[-1] < best_makespan:
                    best_makespan = comp[-1]
                    best_pos = pos
            # move job to best_pos
            jobs.pop(i)
            jobs.insert(best_pos, jobs[i])  # jobs[i] was the one we moved
        return jobs