def solve(**kwargs):
    N = kwargs.get("N", 0)
    K = kwargs.get("K", 1)
    time_limit = kwargs.get("time_limit", 0.0)
    tasks = kwargs.get("tasks", {})
    arcs = kwargs.get("arcs", {})
    if N == 0:
        return {"crews": []}
    sorted_tasks = sorted(tasks.keys(), key=lambda t: tasks[t][0])
    crews = [[] for _ in range(K)]
    for i, task in enumerate(sorted_tasks):
        crews[i % K].append(task)
    feasible = True
    total_cost = 0.0
    for crew in crews:
        if not crew:
            continue
        for j in range(len(crew) - 1):
            if tasks[crew[j]][1] > tasks[crew[j + 1]][0]:
                feasible = False
                break
        if not feasible:
            break
        duty = tasks[crew[-1]][1] - tasks[crew[0]][0]
        if duty > time_limit:
            feasible = False
            break
        for j in range(len(crew) - 1):
            from_t = crew[j]
            to_t = crew[j + 1]
            if (from_t, to_t) not in arcs:
                feasible = False
                break
            total_cost += arcs[(from_t, to_t)]
    if not feasible:
        crews = [[t] for t in sorted_tasks[:K]]
    return {"crews": crews}