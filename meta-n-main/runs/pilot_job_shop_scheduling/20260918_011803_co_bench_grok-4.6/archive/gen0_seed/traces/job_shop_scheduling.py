def solve(**kwargs):
    """
    Solves a single job shop scheduling test case.

    Input:
        - n_jobs (int): Number of jobs.
        - n_machines (int): Number of machines (and operations per job).
        - times (list of list of int): A 2D list of processing times for each operation.
          Dimensions: n_jobs x n_machines.
        - machines (list of list of int): A 2D list specifying the machine assignment for each operation.
          Dimensions: n_jobs x n_machines. Note machine is 1-indexed.

    Output:
        solution (dict): A dictionary containing:
            - start_times (list of list of int): A 2D list of start times for each operation.
              Dimensions: n_jobs x n_machines.

            Each start time must be a non-negative integer, and the schedule must respect the following constraints:
                (i) Sequential processing: For each job, an operation cannot start until its preceding operation has finished.
                (ii) Machine exclusivity: For operations assigned to the same machine, their processing intervals must not overlap.

            The evaluation function will use the start_times to compute the makespan and verify the constraints.
    """
    n_jobs = kwargs["n_jobs"]
    n_machines = kwargs["n_machines"]
    times = kwargs["times"]
    machines = kwargs["machines"]

    # Initialize start times
    start_times = [[0 for _ in range(n_machines)] for _ in range(n_jobs)]

    # Current available time for each machine (0-indexed)
    machine_avail = [0] * n_machines

    # Schedule operations job by job
    for j in range(n_jobs):
        prev_finish = 0
        for i in range(n_machines):
            m = machines[j][i] - 1  # convert to 0-indexed
            t = times[j][i]
            # Start time is max of previous operation finish and machine availability
            start = max(prev_finish, machine_avail[m])
            start_times[j][i] = start
            # Update previous finish for next operation in job
            prev_finish = start + t
            # Update machine availability
            machine_avail[m] = start + t

    solution = {"start_times": start_times}
    return solution