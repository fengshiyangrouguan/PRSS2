if task.task_id == "flow_shop_scheduling":
    additional_context = """You are solving the m-machine flow-shop scheduling problem. Use the provided neh_heuristic from solver_lib to compute the job sequence. Never implement DP or other algorithms; always call neh_heuristic(n, m, matrix). Return {'job_sequence': list_of_job_indices_0based}. The returned sequence must be a valid permutation (all jobs 0..n-1 exactly once)."""
else:
    additional_context = ""