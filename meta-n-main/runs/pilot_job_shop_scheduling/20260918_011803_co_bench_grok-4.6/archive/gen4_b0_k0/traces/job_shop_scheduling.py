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

    # List of all operations as (job, op_index)
    ops = [(j, i) for j in range(n_jobs) for i in range(n_machines)]
    # Sort by processing time descending (LPT)
    ops.sort(key=lambda op: times[op[0]][op[1]], reverse=True)

    # Machine free times (earliest available time)
    machine_free = [0] * n_machines
    # Start times (to be filled)
    start_times = [[0 for _ in range(n_machines)] for _ in range(n_jobs)]
    # Predecessor finish times (for each op)
    pred_finish = [0] * (n_jobs * n_machines)

    # Ready queue: max-heap of (-proc, job, op)
    ready = []
    # Initialize with first operations of each job
    for j in range(n_jobs):
        if n_machines > 0:
            ready.append((-times[j][0], j, 0))
    heapq.heapify(ready)

    scheduled = set()
    while ready:
        _, j, i = heapq.heappop(ready)
        op_idx = j * n_machines + i
        if op_idx in scheduled:
            continue
        scheduled.add(op_idx)
        proc = times[j][i]
        m = machines[j][i] - 1  # 0-index machine
        # Earliest start time: max(machine free, pred finish)
        start = max(machine_free[m], pred_finish[op_idx])
        start_times[j][i] = start
        finish = start + proc
        machine_free[m] = finish
        pred_finish[op_idx] = finish
        # Release successor if exists
        if i + 1 < n_machines:
            succ_j, succ_i = j, i + 1
            succ_proc = times[succ_j][succ_i]
            heapq.heappush(ready, (-succ_proc, succ_j, succ_i))

    solution = {"start_times": start_times}
    return solution