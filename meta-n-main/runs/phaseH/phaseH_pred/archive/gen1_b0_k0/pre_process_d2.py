desc = (getattr(task, "description", "") or "").lower()
meta = getattr(task, "metadata", {}) or {}

if "assignment" in desc or ("cost_matrix" in desc and "total_cost" in desc):
    additional_context = """
You are improving an existing assignment-problem solver. Do NOT replace the good exact behavior for small instances: keep the Hungarian algorithm for n up to roughly 400-500 because it gets optimal scores there.

Critical issue: the current O(n^3) Hungarian implementation times out on larger instances such as n=700 and n=3000, causing zero score for those cases. For large n, continuous scoring rewards returning a feasible assignment quickly, even if approximate. Add a size/time guard:
- if n <= about 450: use the existing Hungarian algorithm.
- otherwise: use a fast feasible heuristic and return before the 10s limit.

A robust large-n heuristic:
1. Convert cost_matrix to list-like rows without copying more than necessary.
2. For each row, scan all columns once to find its best and second-best columns/costs; compute regret = second_best - best.
3. Process rows in descending regret, so rows with few good choices get priority.
4. Maintain a boolean/free set of unused columns.
5. For each row in that order, if its best column is free, take it; otherwise scan that row for the cheapest still-free column.
6. Build 1-indexed (row, column) tuples and exact reported total_cost from the chosen entries.
This is O(n^2), feasible for n=3000, and should greatly outperform a timeout.

A helper named fast_greedy_assignment is available; call it for large n if you want a compact implementation.
"""
else:
    additional_context = ""