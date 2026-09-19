def solve(**kwargs):
    N = kwargs.get("N", 0)
    K = kwargs.get("K", 1)
    time_limit = kwargs.get("time_limit", 0.0)
    tasks = kwargs.get("tasks", {})
    arcs = kwargs.get("arcs", {})
    if N == 0:
        return {"crews": []}
    task_list = sorted(tasks.keys(), key=lambda t: tasks[t][0])
    possible = {}
    for i in task_list:
        possible[i] = []
        fi, _ = tasks[i]
        for j in task_list:
            if i == j:
                continue
            fj, _ = tasks[j]
            if fj <= fi:
                continue
            if (i, j) in arcs:
                possible[i].append((j, arcs[(i, j)]))
    crews = [[] for _ in range(K)]
    current_costs = [0.0] * K
    crew_last = [None] * K
    assigned = set()
    for task in task_list:
        best_crew = -1
        best_cost = float("inf")
        for c in range(K):
            if crew_last[c] is None:
                if tasks[task][1] - tasks[task][0] > time_limit:
                    continue
                if best_crew == -1 or current_costs[c] < best_cost:
                    best_crew = c
                    best_cost = current_costs[c]
                continue
            last = crew_last[c]
            can = False
            cost = 0.0
            for nxt, cst in possible[last]:
                if nxt == task:
                    can = True
                    cost = cst
                    break
            if can:
                new_cost = current_costs[c] + cost
                if new_cost < best_cost:
                    best_crew = c
                    best_cost = new_cost
        if best_crew == -1:
            best_crew = 0
        crews[best_crew].append(task)
        if crew_last[best_crew] is None:
            crew_last[best_crew] = task
            current_costs[best_crew] = 0.0
        else:
            for nxt, cst in possible[crew_last[best_crew]]:
                if nxt == task:
                    current_costs[best_crew] += cst
                    break
        crew_last[best_crew] = task
        assigned.add(task)
    for c in range(K):
        if not crews[c]:
            continue
        first = crews[c][0]
        last = crews[c][-1]
        duty = tasks[last][1] - tasks[first][0]
        if duty > time_limit:
            return {"crews": []}
    return {"crews": crews}