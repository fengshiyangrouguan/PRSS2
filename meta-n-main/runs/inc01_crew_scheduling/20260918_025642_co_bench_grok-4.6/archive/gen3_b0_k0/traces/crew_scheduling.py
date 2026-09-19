from solver_lib import greedy_crew_assignment

def solve(**kwargs):
    N = kwargs.get("N")
    K = kwargs.get("K")
    time_limit = kwargs.get("time_limit")
    tasks = kwargs.get("tasks")
    arcs = kwargs.get("arcs")
    sorted_tasks = sorted(tasks.items(), key=lambda x: x[1][0])
    task_ids = [task_id for task_id, _ in sorted_tasks]
    crews = greedy_crew_assignment(task_ids, K, time_limit, tasks, arcs)
    return {"crews": crews}