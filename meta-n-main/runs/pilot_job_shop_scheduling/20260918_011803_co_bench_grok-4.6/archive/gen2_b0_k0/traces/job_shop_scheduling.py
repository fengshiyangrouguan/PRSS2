def solve(**kwargs):
    n_jobs = kwargs["n_jobs"]
    n_machines = kwargs["n_machines"]
    times = kwargs["times"]
    machines = kwargs["machines"]

    # Compute total processing time for each job and sort jobs by LPT (descending)
    job_totals = [sum(times[j]) for j in range(n_jobs)]
    sorted_jobs = sorted(range(n_jobs), key=lambda j: job_totals[j], reverse=True)

    # Initialize start times (all zero) and machine availability
    start_times = [[0 for _ in range(n_machines)] for _ in range(n_jobs)]
    machine_avail = [0] * n_machines

    # Schedule each job in LPT order, operations in original sequence
    for job in sorted_jobs:
        prev_finish = 0
        for i in range(n_machines):
            m = machines[job][i] - 1  # convert to 0-indexed
            t = times[job][i]
            start = max(prev_finish, machine_avail[m])
            start_times[job][i] = start
            machine_avail[m] = start + t
            prev_finish = start + t

    solution = {"start_times": start_times}
    return solution