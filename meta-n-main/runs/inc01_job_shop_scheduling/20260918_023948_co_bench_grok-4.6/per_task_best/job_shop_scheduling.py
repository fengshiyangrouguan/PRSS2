def solve(**kwargs):
    n_jobs = kwargs["n_jobs"]
    n_machines = kwargs["n_machines"]
    times = kwargs["times"]
    machines = kwargs["machines"]

    # Compute suffix sums for each job (longest path to end)
    suffix = [[0] * n_machines for _ in range(n_jobs)]
    for j in range(n_jobs):
        for m in range(n_machines - 1, -1, -1):
            suffix[j][m] = times[j][m] + (suffix[j][m + 1] if m + 1 < n_machines else 0)

    # List of operations: (job, op_index)
    operations = [(j, m) for j in range(n_jobs) for m in range(n_machines)]
    operations.sort(key=lambda jm: suffix[jm[0]][jm[1]], reverse=True)

    # Initialize start times and machine finish times
    start_times = [[0] * n_machines for _ in range(n_jobs)]
    machine_finish = [0] * (n_machines + 1)

    for j, m in operations:
        mach = machines[j][m]
        # Earliest start respecting job predecessor
        if m == 0:
            s = 0
        else:
            s = start_times[j][m - 1] + times[j][m - 1]
        # Earliest start respecting machine
        s = max(s, machine_finish[mach])
        start_times[j][m] = s
        machine_finish[mach] = s + times[j][m]

    solution = {"start_times": start_times}
    return solution