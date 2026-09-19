import heapq
from solver_lib import jss_local_search

def solve(**kwargs):
    n_jobs = kwargs["n_jobs"]
    n_machines = kwargs["n_machines"]
    times = kwargs["times"]
    machines = kwargs["machines"]

    # Precompute critical path lengths (remaining processing time to end of job)
    crit = [[0] * n_machines for _ in range(n_jobs)]
    for j in range(n_jobs):
        for i in range(n_machines - 1, -1, -1):
            crit[j][i] = times[j][i] + (crit[j][i + 1] if i + 1 < n_machines else 0)

    # List of all operations: (job, op)
    ops = [(j, i) for j in range(n_jobs) for i in range(n_machines)]

    # Priority queue: (priority, op_id) where priority = -crit (max-heap via min-heap)
    ready = []
    indeg = [0] * (n_jobs * n_machines)
    for j in range(n_jobs):
        for i in range(1, n_machines):
            prev = j * n_machines + (i - 1)
            indeg[j * n_machines + i] = 1
    for j in range(n_jobs):
        op_id = j * n_machines
        heapq.heappush(ready, (-crit[j][0], op_id))

    start_times = [[0 for _ in range(n_machines)] for _ in range(n_jobs)]
    job_next_start = [0] * n_jobs
    machine_finish = [0] * n_machines

    while ready:
        _, op_id = heapq.heappop(ready)
        j = op_id // n_machines
        i = op_id % n_machines
        mach = machines[j][i] - 1
        start = max(job_next_start[j], machine_finish[mach])
        finish = start + times[j][i]
        start_times[j][i] = start
        job_next_start[j] = finish
        machine_finish[mach] = finish
        next_op_id = j * n_machines + (i + 1)
        if i + 1 < n_machines:
            indeg[next_op_id] -= 1
            if indeg[next_op_id] == 0:
                heapq.heappush(ready, (-crit[j][i + 1], next_op_id))

    # Apply local search for improvement
    jss_local_search(start_times, times, machines, max_passes=3)

    solution = {"start_times": start_times}
    return solution