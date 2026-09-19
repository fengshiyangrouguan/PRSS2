def pre_process(task, outer_context):
    n = task.metadata.get("n", 0)
    m = task.metadata.get("m", 0)
    if m == 2:
        additional_context = """When m == 2 invoke Johnson's rule (group jobs by a_j < b_j or a_j > b_j, sort group1 ascending on a, group2 descending on b, concatenate; for n <= 1 return identity). For all other m implement the 3-seed NEH insertion heuristic exactly as described previously: sort jobs by total processing time, enumerate all 6 permutations of the first 3 as seed, pick the best makespan seed, then iteratively insert each remaining job (sorted order) into the single best position by evaluating makespan on each of the O(n) positions. Always return job_sequence as flat list [1..n]. Handle n <= 3 by full enumeration. Use only standard library; keeps makespan evaluator identical to baseline."""
    elif n <= 20:
        additional_context = """For small n (<=20) use the 3-seed NEH insertion heuristic: sort jobs by total processing time descending, enumerate all 6 permutations of first 3, pick best makespan seed, then iteratively insert each remaining job (in sorted order) into the single best position w.r.t. makespan. Handle n<=3 as special case by full permutation. Always return job_sequence as flat list of integers [1..n] with no duplicates or omissions. Use only standard library; keep makespan evaluation identical to baseline."""
    else:
        additional_context = """For large n (>20) and m>2 use simple heuristic: sort jobs by total processing time descending (O(n log n)). This avoids O(n^3 m) timeout on tai200/500 instances while matching or beating prior NEH3 on those. Always return job_sequence as flat list of integers [1..n] with no duplicates or omissions. Use only standard library."""
    return additional_context