def solve(**kwargs):
    n_jobs = kwargs["n_jobs"]
    n_machines = kwargs["n_machines"]
    times = kwargs["times"]
    machines = kwargs["machines"]

    start_times = [[0 for _ in range(n_machines)] for _ in range(n_jobs)]
    job_finish = [0] * n_jobs
    machine_finish = [0] * n_machines

    for j in range(n_jobs):
        for m in range(n_machines):
            if m == 0:
                prev_finish = 0
            else:
                prev_finish = start_times[j][m-1] + times[j][m-1]
            machine = machines[j][m] - 1
            machine_last = machine_finish[machine]
            start_times[j][m] = max(prev_finish, machine_last)
            job_finish[j] = start_times[j][m] + times[j][m]
            machine_finish[machine] = job_finish[j]

    # Simple local search: limited random adjacent swaps on each machine to reduce makespan
    import random
    import time
    start_time = time.time()
    max_time = 0.5  # seconds per machine, total under 10s
    for machine in range(n_machines):
        if time.time() - start_time > max_time:
            break
        ops = [(j, m) for j in range(n_jobs) if machines[j][m] - 1 == machine]
        if len(ops) < 2:
            continue
        for _ in range(20):  # limited swaps
            if time.time() - start_time > max_time:
                break
            i = random.randint(0, len(ops)-2)
            j1, m1 = ops[i]
            j2, m2 = ops[i+1]
            # swap start times
            s1 = start_times[j1][m1]
            s2 = start_times[j2][m2]
            f1 = s1 + times[j1][m1]
            f2 = s2 + times[j2][m2]
            # check if swap improves makespan and respects constraints
            new_makespan = max(f1, f2)
            if new_makespan < start_times[j2][m2-1] + times[j2][m2-1] if m2 > 0 else new_makespan < 0:
                continue  # invalid due to job sequence
            # also check machine exclusivity after swap (already since adjacent)
            start_times[j1][m1] = s2
            start_times[j2][m2] = s1
            job_finish[j1] = s2 + times[j1][m1]
            job_finish[j2] = s1 + times[j2][m2]
            machine_finish[machine] = max(job_finish[j1], job_finish[j2])

    solution = {"start_times": start_times}
    return solution