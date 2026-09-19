def crew_scheduling(tasks: dict, arcs: dict, K: int, time_limit: float) -> dict:
    """Greedy crew assignment for crew scheduling with transition costs.

    Args:
        tasks: dict mapping task_id to (start, end) tuples.
        arcs: dict mapping (prev, next) -> cost for allowed transitions.
        K: int number of crews.
        time_limit: float max allowed duty length per crew.

    Returns:
        dict with key 'crews' holding list of K lists of task_ids in assignment order,
        or empty list if infeasible.
    """
    if not tasks:
        return {"crews": []}
    task_list = sorted(tasks.keys(), key=lambda t: tasks[t][0])
    # Build feasible transitions (later tasks only, no overlap)
    possible = {}
    for i in task_list:
        possible[i] = []
        fi, ei = tasks[i]
        for j in task_list:
            if i == j:
                continue
            fj, _ = tasks[j]
            if fj <= fi:
                continue
            if ei > fj:  # overlap check
                continue
            if (i, j) in arcs:
                possible[i].append((j, arcs[(i, j)]))
    # Greedy assignment
    crews = [[] for _ in range(K)]
    crew_last = [None] * K
    current_costs = [0.0] * K
    assigned = set()
    for task in task_list:
        best_crew = -1
        best_cost = float("inf")
        for c in range(K):
            last = crew_last[c]
            if last is None:
                start, end = tasks[task]
                if end - start > time_limit:
                    continue
                can = True
                cost = 0.0
            else:
                can = False
                cost = 0.0
                for nxt, cst in possible[last]:
                    if nxt == task:
                        can = True
                        cost = cst
                        break
                if not can or tasks[last][1] > tasks[task][0]:
                    continue
            new_cost = current_costs[c] + cost
            if new_cost < best_cost:
                best_crew = c
                best_cost = new_cost
        if best_crew == -1:
            return {"crews": []}
        crews[best_crew].append(task)
        if crew_last[best_crew] is None:
            crew_last[best_crew] = task
            current_costs[best_crew] = 0.0
        else:
            current_costs[best_crew] = best_cost
        crew_last[best_crew] = task
        assigned.add(task)
    if len(assigned) != len(task_list):
        return {"crews": []}
    # Final duty length check
    for c in range(K):
        crew = crews[c]
        if not crew:
            continue
        first = task_list[0] if crew[0] is None else crew[0]
        last = crew[-1]
        duty = tasks[last][1] - tasks[first][0]
        if duty > time_limit:
            return {"crews": []}
    return {"crews": crews}