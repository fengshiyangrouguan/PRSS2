import heapq

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

    start_times = [[0 for _ in range(n_machines)] for _ in range(n_jobs)]
    machine_free = [0] * n_machines
    job_ready = [0] * n_jobs

    pq = []
    for j in range(n_jobs):
        heapq.heappush(pq, (-times[j][0], j, 0))

    while pq:
        _, j, i = heapq.heappop(pq)
        start = max(job_ready[j], machine_free[machines[j][i] - 1])
        start_times[j][i] = start
        finish = start + times[j][i]
        machine_free[machines[j][i] - 1] = finish
        job_ready[j] = finish
        if i + 1 < n_machines:
            heapq.heappush(pq, (-times[j][i + 1], j, i + 1))

    solution = {"start_times": start_times}
    return solution