def solve(**kwargs):
    """
    Solves the crew scheduling problem.

    The problem consists of assigning each task (with a defined start and finish time) to exactly one crew,
    such that:
      - The tasks within each crew are executed in non-overlapping order.
      - For every consecutive pair of tasks in a crew’s schedule, a valid transition arc exists (with an associated cost).
      - The overall duty time (finish time of the last task minus start time of the first task) does not exceed the specified time limit.
      - Exactly K crews are used.

    Input kwargs (for one case):
      - N (int): Number of tasks.
      - K (int): Maximum number of crews to be used.
      - time_limit (float): Maximum allowed duty time.
      - tasks (dict): Dictionary mapping task ID (1 to N) to a tuple (start_time, finish_time).
      - arcs (dict): Dictionary mapping (from_task, to_task) pairs to transition cost.

    Evaluation metric:
      - If all constraints are met (no task overlap, valid transition arcs, duty time within the limit, and exactly K crews used), the score is the sum of transition costs across all crews.
      - If any constraint is violated, the solution is infeasible and receives no score.
      - A lower score indicates a more cost-effective solution.

    Returns:
      dict: A dictionary with one key "crews", whose value is a list of lists. Each inner list is a sequence of task IDs (integers)
            representing one crew’s schedule.
    """
    from solver_lib import crew_scheduling
    return crew_scheduling(kwargs.get("tasks"), kwargs.get("arcs"), kwargs.get("K"), kwargs.get("time_limit"))