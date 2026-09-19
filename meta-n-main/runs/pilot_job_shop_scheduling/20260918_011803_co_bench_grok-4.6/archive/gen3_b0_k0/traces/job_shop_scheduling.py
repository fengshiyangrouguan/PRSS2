import heapq

def solve(**kwargs):
    n_jobs = kwargs["n_jobs"]
    n_machines = kwargs["n_machines"]
    times = kwargs["times"]
    machines = kwargs["machines"]
    
    start_times = [[0 for _ in range(n_machines)] for _ in range(n_jobs)]
    machine_avail = [0] * n_machines
    job_finish = [0] * n_jobs
    ready = []
    next_op = [0] * n_jobs
    
    # Initialize ready queue with first operations of each job, prioritized by LPT
    for j in range(n_jobs):
        i = 0
        p = times[j][i]
        heapq.heappush(ready, (-p, j, i))
    
    while ready:
        _, j, i = heapq.heappop(ready)
        p = times[j][i]
        m = machines[j][i] - 1
        pred_finish = job_finish[j] if i > 0 else 0
        start = max(pred_finish, machine_avail[m])
        start_times[j][i] = start
        finish = start + p
        job_finish[j] = finish
        machine_avail[m] = finish
        if i + 1 < n_machines:
            next_j = j
            next_i = i + 1
            next_p = times[next_j][next_i]
            heapq.heappush(ready, (-next_p, next_j, next_i))
    
    solution = {"start_times": start_times}
    return solution