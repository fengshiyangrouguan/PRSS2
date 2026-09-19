additional_context = (
    "After generating the initial solution with the critical-path greedy, "
    "call the jss_local_search helper (provided in solver_lib) once with "
    "max_passes=3. This will swap adjacent operations on the same machine "
    "when the swap reduces the makespan, raising the continuous score on "
    "every instance."
)