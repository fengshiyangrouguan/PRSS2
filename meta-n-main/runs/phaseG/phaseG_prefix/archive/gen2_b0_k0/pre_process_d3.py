additional_context = """
For large dense assignment_problem instances, the current bottleneck is not validity but quality:
the regret-greedy heuristic is feasible but can be very poor on n=3000 structured/planted cases.

Use this size-based strategy:
- If n <= about 400, keep using the exact Hungarian implementation; it is already perfect there.
- If n > 400, do NOT use pure greedy as the main method. Call the solver_lib helper
  auction_assignment(cost_matrix, n, time_limit=8.5). It returns (total_cost, assignment)
  using a time-limited auction/primal-dual method and should give much better large-instance
  costs than one-pass greedy while still avoiding Hungarian timeouts.
- Return {'total_cost': total_cost, 'assignment': assignment}.
- The assignment must remain 1-indexed (row, column) pairs.
"""