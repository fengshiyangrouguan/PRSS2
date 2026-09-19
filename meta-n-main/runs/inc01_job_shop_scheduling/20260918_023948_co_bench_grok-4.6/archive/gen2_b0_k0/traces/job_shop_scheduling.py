import time
import math

def solve(**kwargs):
    n_jobs = kwargs["n_jobs"]
    n_machines = kwargs["n_machines"]
    times = kwargs["times"]
    machines = kwargs["machines"]
    
    # Longest-path heuristic: compute earliest start times respecting job precedence and machine availability
    start_times = [[0] * n_machines for _ in range(n_jobs)]
    machine_finish = [0] * n_machines
    for j in range(n_jobs):
        prev_finish = 0
        for m in range(n_machines):
            op_machine = machines[j][m] - 1
            start = max(prev_finish, machine_finish[op_machine])
            start_times[j][m] = start
            finish = start + times[j][m]
            machine_finish[op_machine] = finish
            prev_finish = finish
    
    # Local search: repeatedly swap consecutive operations on the same machine if it reduces makespan
    best_start_times = [row[:] for row in start_times]
    best_makespan = max(machine_finish)
    improved = True
    max_passes = 100
    pass_count = 0
    while improved and pass_count < max_passes:
        improved = False
        pass_count += 1
        for j in range(n_jobs):
            for m in range(n_machines - 1):
                if machines[j][m] == machines[j][m + 1]:
                    # Swap the two consecutive operations on the same machine
                    s1 = start_times[j][m]
                    s2 = start_times[j][m + 1]
                    t1 = times[j][m]
                    t2 = times[j][m + 1]
                    # Temporarily apply swap and check effect on makespan
                    # Compute new finish times for both ops and affected machines
                    # This requires tracking all machine finishes before and after swap
                    # For simplicity, we evaluate the swap by simulating the change
                    # Since n small, we can recompute full schedule after swap
                    new_start = [row[:] for row in start_times]
                    new_start[j][m] = s2
                    new_start[j][m + 1] = s1
                    # Recompute machine finishes with the swapped schedule
                    temp_machine_finish = [0] * n_machines
                    temp_makespan = 0
                    for jj in range(n_jobs):
                        prev_f = 0
                        for mm in range(n_machines):
                            op_m = machines[jj][mm] - 1
                            st = max(prev_f, temp_machine_finish[op_m])
                            if jj == j and mm == m:
                                st = s2
                            elif jj == j and mm == m + 1:
                                st = s1
                            fin = st + times[jj][mm]
                            temp_machine_finish[op_m] = fin
                            prev_f = fin
                    new_makespan = max(temp_machine_finish)
                    if new_makespan < best_makespan:
                        best_makespan = new_makespan
                        best_start_times = [row[:] for row in new_start]
                        improved = True
    # Return the best found schedule
    solution = {"start_times": best_start_times}
    return solution