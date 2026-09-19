def solve(**kwargs):
    N = kwargs.get("N")
    K = kwargs.get("K")
    time_limit = kwargs.get("time_limit")
    tasks = kwargs.get("tasks", {})
    arcs = kwargs.get("arcs", {})
    task_list = [(tid, tasks[tid][0], tasks[tid][1]) for tid in range(1, N + 1)]
    task_list.sort(key=lambda x: x[1])
    crews = [[] for _ in range(K)]
    crew_last_finish = [float("-inf")] * K
    crew_min_start = [float("inf")] * K
    crew_max_finish = [float("-inf")] * K
    assigned = set()
    for tid, start, finish in task_list:
        duration = finish - start
        if duration > time_limit:
            return {"crews": []}
        best_crew = -1
        best_cost = float("inf")
        for c in range(K):
            if crews[c]:
                last_task = crews[c][-1]
                key = (last_task, tid)
                if key not in arcs:
                    continue
                cost = arcs[key]
                if cost >= best_cost:
                    continue
                if crew_last_finish[c] > start:
                    continue
                if crew_max_finish[c] - crew_min_start[c] + duration > time_limit:
                    continue
                best_crew = c
                best_cost = cost
            else:
                if start < crew_min_start[c]:
                    crew_min_start[c] = start
                if finish > crew_max_finish[c]:
                    crew_max_finish[c] = finish
                duty = crew_max_finish[c] - crew_min_start[c]
                if duty > time_limit:
                    continue
                if 0 < best_cost:
                    best_crew = c
                    best_cost = 0
        if best_crew == -1:
            return {"crews": []}
        crews[best_crew].append(tid)
        crew_last_finish[best_crew] = finish
        assigned.add(tid)
    if len(assigned) < N:
        return {"crews": []}
    return {"crews": crews}