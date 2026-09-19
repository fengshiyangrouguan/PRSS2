if "flow_shop_scheduling" in task.description.lower():
    additional_context = "Use the injected makespan helper from solver_lib. Implement the NEH heuristic (sort jobs by total processing time descending; start with first two jobs in best order then insert each remaining job into the position minimizing makespan). Return 1-based job sequence. This replaces any permutation or sequential fallback."
else:
    additional_context = ""