additional_context = """
For the assignment problem, preserve the existing Hungarian algorithm for small instances where it finishes quickly, but do not use O(n^3) Hungarian on large n. The current failure mode is timeout on n around 700+ and especially 3000.

Recommended strategy:
- Convert cost_matrix to a Python list/list-like once.
- If n <= about 450, use the exact Hungarian algorithm already present.
- If n is larger, return a feasible complete assignment quickly instead of timing out.
- You may call fast_large_assignment(cost_matrix, n, time_limit=8.5), which returns {"total_cost": ..., "assignment": [(row, col), ...]} with 1-indexed rows/columns.
- A nonoptimal feasible solution is much better than a timeout because scoring is continuous.
- Ensure every row and every column appears exactly once in the returned assignment.
"""