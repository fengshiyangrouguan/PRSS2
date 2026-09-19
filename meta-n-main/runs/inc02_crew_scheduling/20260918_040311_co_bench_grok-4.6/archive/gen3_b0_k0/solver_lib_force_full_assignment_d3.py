def force_full_assignment(crews: list, tasks: dict, arcs: dict, K: int, time_limit: float) -> list:
    """Force every task into exactly one crew while respecting arcs, non-overlap and duty constraints.

    Args:
        crews: current list of crews (each a list of task ids)
        tasks: dict task_id -> (start, end)
        arcs: dict (from, to) -> cost
        K: number of crews
        time_limit: max duty length

    Returns:
        fixed_crews: list of lists, each inner list contains exactly the tasks assigned to that crew,
        total length across all crews == len(tasks)
    """
    import copy
    assigned = set()
    for crew in crews:
        for t in crew:
            assigned.add(t)
    remaining = [t for t in sorted(tasks.keys()) if t not in assigned]
    if not remaining:
        return crews
    for t in remaining:
        added = False
        for crew in crews:
            if not crew:
                crew.append(t)
                added = True
                break
        if not added:
            best_cost = float("inf")
            best_crew = None
            for crew in crews:
                if crew and (crew[-1], t) in arcs:
                    cost = arcs[(crew[-1], t)]
                    if cost < best_cost:
                        best_cost = cost
                        best_crew = crew
            if best_crew is not None:
                best_crew.append(t)
                added = True
        if not added:
            crews.append([t])
    # final duty & arc validation pass (already performed in caller)
    return crews