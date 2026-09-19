if "NEH" in outer_context:
    additional_context = "Use the provided NEH solver_lib function as your primary solver; ignore permutations and timeouts."
else:
    additional_context = "Implement the NEH heuristic from solver_lib to avoid timeouts on large instances."