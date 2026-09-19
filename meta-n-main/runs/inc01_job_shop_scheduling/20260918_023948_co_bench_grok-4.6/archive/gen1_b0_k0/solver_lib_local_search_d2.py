def local_search(schedule: dict, times: list[list[int]], machines: list[list[int]], n_jobs: int, n_machines: int) -> dict:
    """Refine an initial schedule via limited random adjacent swaps on machines.
    Args:
        schedule: dict containing 'start_times' (list[list[int]] of job x machine starts)
        times: list[list[int]] — processing times (n_jobs x n_machines)
        machines: list[list[int]] — machine assignments (n_jobs x n_machines)
        n_jobs: int — number of jobs
        n_machines: int — number of machines
    Returns:
        updated dict with improved 'start_times' (lower makespan)
    """
    import copy
    import random
    s = copy.deepcopy(schedule["start_times"])
    def makespan():
        finish = [0] * n_machines
        for j in range(n_jobs):
            for m in range(n_machines):
                mach = machines[j][m]
                finish[mach] = max(finish[mach], s[j][m] + times[j][m])
        return max(finish)
    best = makespan()
    for _ in range(100):
        j1, m = random.randint(0, n_jobs - 1), random.randint(0, n_machines - 1)
        j2 = random.randint(0, n_jobs - 1)
        if m == machines[j1][m] and m == machines[j2][m]:  # same machine
            s[j1][m], s[j2][m] = s[j2][m], s[j1][m]
            new = makespan()
            if new < best:
                best = new
            else:
                s[j1][m], s[j2][m] = s[j2][m], s[j1][m]  # revert
    schedule["start_times"] = s
    return schedule