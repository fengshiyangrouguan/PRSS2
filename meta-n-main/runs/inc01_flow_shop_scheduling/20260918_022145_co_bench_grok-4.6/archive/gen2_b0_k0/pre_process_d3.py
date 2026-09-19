if task.task_id == "flow_shop_scheduling":
    additional_context = """You are solving the m-machine flow-shop scheduling problem (n jobs, m machines). The optimal makespan is NP-hard for m>2, so use Johnson's rule as a fast heuristic (optimal for m=2). Implement a Johnson's heuristic or NEH insertion heuristic in the solver; only fall back to the provided DP for n<=12. Never use lists vs tuples in comparisons. Return {'job_sequence': list_of_job_indices_0based}. The returned sequence must be a valid permutation (all jobs 0..n-1 exactly once)."""
else:
    additional_context = ""