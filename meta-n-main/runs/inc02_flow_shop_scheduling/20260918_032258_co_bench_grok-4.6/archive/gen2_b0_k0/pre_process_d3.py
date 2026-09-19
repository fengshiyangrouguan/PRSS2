if "flow_shop_scheduling" in task.description.lower():
    additional_context = "Ignore any permutation or fallback logic. Use neh_heuristic from solver_lib for all n/m. Maintain C matrix incrementally when inserting (reuse makespan). Return 1-based job sequence. Handle n=0/m=0 as empty/trivial."
else:
    additional_context = ""