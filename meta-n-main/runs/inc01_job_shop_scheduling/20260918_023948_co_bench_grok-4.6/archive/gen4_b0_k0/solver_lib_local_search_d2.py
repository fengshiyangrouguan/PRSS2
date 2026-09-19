def local_search(start_times: list, machines: list, times: list, n_jobs: int, n_machines: int) -> dict:
    """Improve the schedule by swapping consecutive operations on each machine if the makespan decreases.

    Args:
        start_times: 2D list where start_times[j][m] is the start time of operation (job j, machine m)
        machines: 2D list where machines[j][m] is the machine assigned to operation (job j, machine m)
        times: 2D list where times[j][m] is the processing time of operation (job j, machine m)
        n_jobs: number of jobs
        n_machines: number of machines

    Returns:
        dict with updated start_times that have been locally improved.
    """
    import copy
    improved = copy.deepcopy(start_times)
    machine_finish = [0.0] * n_machines
    for j in range(n_jobs):
        for m in range(n_machines - 1):
            op1_mach = machines[j][m]
            op2_mach = machines[j][m + 1]
            if op1_mach == op2_mach:
                # swap on same machine: does it reduce makespan?
                old_s1 = improved[j][m]
                old_s2 = improved[j][m + 1]
                p1 = times[j][m]
                p2 = times[j][m + 1]
                # tentative new times
                new_s1 = old_s2 - p1
                new_s2 = old_s1 + p1
                # check if new_s1 >= machine_finish[op1_mach] and new_s2 >= machine_finish[op1_mach]
                if new_s1 >= machine_finish[op1_mach] and new_s2 >= machine_finish[op1_mach]:
                    # also respect job predecessors (already true since consecutive)
                    improved[j][m] = new_s1
                    improved[j][m + 1] = new_s2
                    machine_finish[op1_mach] = max(machine_finish[op1_mach], new_s1 + p1, new_s2 + p2)
    return {"start_times": improved}