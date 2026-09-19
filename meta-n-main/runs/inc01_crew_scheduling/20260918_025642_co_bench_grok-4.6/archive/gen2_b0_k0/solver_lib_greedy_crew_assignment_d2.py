def greedy_crew_assignment(N: int, K: int, time_limit: float, tasks: dict, arcs: dict) -> dict:
    """Greedy assignment of N tasks to K crews.

    Args:
        N: total number of tasks to cover
        K: number of crews
        time_limit: maximum duration any crew may cover
        tasks: dict mapping task_id to [start_time, end_time] float pair
        arcs: dict mapping (prev_task, next_task) -> connection_cost float

    Returns:
        {"crews": list of lists}, where each inner list is the ordered task_ids for one crew
        (empty lists if any task could not be assigned)
    """
    import sys
    # Sort tasks by start time
    task_list = sorted(tasks.keys(), key=lambda t: tasks[t][0])
    crews = [[] for _ in range(K)]
    assigned = set()
    for task in task_list:
        if task in assigned:
            continue
        best_added = sys.float_info.max
        best_crew = -1
        best_prev = None
        for c_idx in range(K):
            crew = crews[c_idx]
            if not crew:  # empty crew - start new
                s, f = tasks[task]
                if f - s > time_limit:
                    continue
                added = 0.0
                if added < best_added:
                    best_added = added
                    best_crew = c_idx
                    best_prev = None
                continue
            last = crew[-1]
            if (last, task) not in arcs:
                continue
            cost = arcs[(last, task)]
            last_f = tasks[last][1]
            if last_f > tasks[task][0]:
                continue
            new_f = max(last_f, tasks[task][1])
            if new_f - tasks[crew[0]][0] > time_limit:
                continue
            added = cost
            if added < best_added:
                best_added = added
                best_crew = c_idx
                best_prev = last
        if best_crew != -1:
            if best_prev is None:
                crews[best_crew] = [task]
            else:
                crews[best_crew].append(task)
            assigned.add(task)
    # Validate coverage
    if len(assigned) != N:
        return {"crews": [[] for _ in range(K)]}
    return {"crews": crews}