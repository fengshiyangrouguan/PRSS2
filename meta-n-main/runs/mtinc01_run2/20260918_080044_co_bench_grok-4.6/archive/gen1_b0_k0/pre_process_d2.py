if "bin packing" in task.description.lower() or "one dimensional" in task.description.lower():
    additional_context = "Use the prepended best_fit_decreasing helper for first-fit-style assignment; it finds the tightest-fitting bin instead of the first one."
else:
    additional_context = ""