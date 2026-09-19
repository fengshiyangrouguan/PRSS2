additional_context = """Upgrade the solver to the best-known heuristic for flow_shop_scheduling (O(n log n), works for all n):
Extract per-job first and last processing times: p1 = matrix[i][0], pm = matrix[i][m-1].
Split jobs into group1 (p1 <= pm) and group2 (p1 > pm).
Sort group1 ascending by p1, group2 descending by pm.
Return concatenated job_sequence (0-based indices +1).
This beats brute force on large tai* instances; use only this logic, ignore permutations entirely."""