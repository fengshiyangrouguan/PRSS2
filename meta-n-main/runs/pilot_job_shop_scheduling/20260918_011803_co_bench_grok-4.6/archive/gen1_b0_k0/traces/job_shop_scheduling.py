import heapq

def solve(**kwargs):
    n_jobs = kwargs["n_jobs"]
    n_machines = kwargs["n_machines"]
    times = kwargs["times"]
    machines = kwargs["machines"]
    operations = [(j, i) for j in range(n_jobs) for i in range(n_machines)]
    operations.sort(key=lambda x: times[x[0]][x[1]], reverse=True)
    ready = []
    machine_avail = [0] * n_machines
    start_times = [[0 for _ in range(n_machines)] for _ in range(n_jobs)]
    scheduled = [[False] * n_machines for _ in range(n_jobs)]
    for job in range(n_jobs):
        p = times[job][0]
        heapq.heappush(ready, (-p, job, 0))
    while ready:
        _, job, op = heapq.heappop(ready)
        if scheduled[job][op]:
            continue
        scheduled[job][op] = True
        if op == 0:
            pred_finish = 0
        else:
            pred_finish = start_times[job][op-1] + times[job][op-1]
        mach = machines[job][op] - 1
        start = max(pred_finish, machine_avail[mach])
        start_times[job][op] = start
        machine_avail[mach] = start + times[job][op]
        if op < n_machines - 1:
            next_op = op + 1
            next_p = times[job][next_op]
            heapq.heappush(ready, (-next_p, job, next_op))
    solution = {"start_times": start_times}
    return solution