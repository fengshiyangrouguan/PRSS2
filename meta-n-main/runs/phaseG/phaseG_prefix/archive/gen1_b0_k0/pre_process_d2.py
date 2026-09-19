additional_context = """
For assignment_problem, do not use pure-Python Hungarian for all sizes: it times out on large dense cases.
Use exact Hungarian only for small/medium n where it can finish comfortably, and for large n switch to a fast feasible heuristic.

Recommended structure:
- Parse n and cost_matrix as before.
- If n <= about 450 or 500, Hungarian is OK.
- If n is larger, call fast_greedy_assignment(cost_matrix, n, time_limit=8.5).
- Return {'total_cost': total_cost, 'assignment': assignment}.
A valid complete assignment with heuristic cost is much better than timing out.
The assignment list should contain 1-based (row, column) pairs and total_cost should be the summed selected costs.
"""