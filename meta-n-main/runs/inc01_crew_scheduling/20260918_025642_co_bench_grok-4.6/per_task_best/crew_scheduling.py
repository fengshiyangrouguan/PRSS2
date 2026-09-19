def solve(**kwargs):
    N = kwargs.get("N")
    K = kwargs.get("K")
    time_limit = kwargs.get("time_limit")
    tasks = kwargs.get("tasks")
    arcs = kwargs.get("arcs")
    if N is None or K is None or time_limit is None or tasks is None or arcs is None:
        return {"crews": [[] for _ in range(K)]}
    task_list = sorted(tasks.keys(), key=lambda t: tasks[t][0])
    crews = [{'last': None, 'start': 0.0, 'finish': 0.0, 'cost': 0.0, 'tasks': []} for _ in range(K)]
    for task in task_list:
        best_added = float('inf')
        best_crew_idx = -1
        for c_idx, crew in enumerate(crews):
            if crew['last'] is None:
                s = tasks[task][0]
                f = tasks[task][1]
                if f - s > time_limit:
                    continue
                added = 0.0
                if added < best_added:
                    best_added = added
                    best_crew_idx = c_idx
                continue
            last = crew['last']
            if (last, task) not in arcs:
                continue
            cost = arcs[(last, task)]
            prev_f = crew['finish']
            if prev_f > tasks[task][0]:
                continue
            s = crew['start']
            new_f = max(crew['finish'], tasks[task][1])
            if new_f - s > time_limit:
                continue
            added = cost
            if added < best_added:
                best_added = added
                best_crew_idx = c_idx
        if best_crew_idx == -1:
            for c_idx, crew in enumerate(crews):
                if crew['last'] is None:
                    s = tasks[task][0]
                    f = tasks[task][1]
                    if f - s > time_limit:
                        continue
                    best_crew_idx = c_idx
                    break
        if best_crew_idx == -1:
            return {"crews": [[] for _ in range(K)]}
        crew = crews[best_crew_idx]
        if crew['last'] is None:
            crew['last'] = task
            crew['start'] = tasks[task][0]
            crew['finish'] = tasks[task][1]
            crew['cost'] = 0.0
            crew['tasks'] = [task]
        else:
            crew['cost'] += arcs[(crew['last'], task)]
            crew['finish'] = max(crew['finish'], tasks[task][1])
            crew['last'] = task
            crew['tasks'].append(task)
    return {"crews": [crew['tasks'] for crew in crews]}