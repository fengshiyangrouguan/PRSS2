if "flow_shop_scheduling" in task.description.lower():
    additional_context = "Use the injected makespan helper from solver_lib. For n<=9 exhaustively try permutations via itertools and pick best makespan. For n>9 apply NEH: (1) compute job total times, (2) sort jobs descending by total, (3) insert each remaining job into best position by calling makespan on the partial sequence. Always return 1-based job_sequence. Handle n=0 gracefully."
else:
    additional_context = ""