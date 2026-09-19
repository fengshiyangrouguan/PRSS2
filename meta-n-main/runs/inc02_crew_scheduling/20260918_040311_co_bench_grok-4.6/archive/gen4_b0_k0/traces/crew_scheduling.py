def solve(**kwargs):
    N = kwargs.get("N")
    K = kwargs.get("K")
    time_limit = kwargs.get("time_limit")
    tasks = kwargs.get("tasks")
    arcs = kwargs.get("arcs")
    return crew_scheduling(tasks, arcs, K, time_limit)