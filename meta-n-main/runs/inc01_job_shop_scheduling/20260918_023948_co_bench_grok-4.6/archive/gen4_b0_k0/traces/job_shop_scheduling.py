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

    # Extract the case parameters
    n_jobs = kwargs["n_jobs"]
    n_machines = kwargs["n_machines"]
    times = kwargs["times"]
    machines = kwargs["machines"]

    # Initialize start times
    start_times = [[0 for _ in range(n_machines)] for _ in range(n_jobs)]

    # Machine free times (earliest available time)
    machine_free = [0] * n_machines

    # Schedule each job one by one
    for j in range(n_jobs):
        current_time = 0
        for m in range(n_machines):
            # Machine index (0-based)
            mk = machines[j][m] - 1
            # Start time is max of job predecessor finish and machine free time
            start = max(current_time, machine_free[mk])
            start_times[j][m] = start
            # Update finish time
            finish = start + times[j][m]
            # Update job current time
            current_time = finish
            # Update machine free time
            machine_free[mk] = finish

    # Build the solution dictionary
    solution = {"start_times": start_times}

    return solution