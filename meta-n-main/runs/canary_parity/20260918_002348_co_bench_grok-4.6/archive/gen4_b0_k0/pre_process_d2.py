if n > 8:
    additional_context = "For n>8, replace the placeholder sequence with the neh_heuristic from solver_lib. For n<=8 keep the existing permutation search as-is (it is fast enough)."
else:
    additional_context = ""