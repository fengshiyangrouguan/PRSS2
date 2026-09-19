if "crew_scheduling" in task.task_id.lower():
    additional_context = "Use the assignment helper from solver_lib to guarantee exactly N unique tasks are placed across crews and fix any count mismatches. Sort tasks by start time. Ensure every task gets a crew (empty or compatible append) and reject if any crew duty exceeds time_limit."
else:
    additional_context = ""