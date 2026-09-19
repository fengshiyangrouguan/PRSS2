def solve(**kwargs):
    n = kwargs['n']
    m = kwargs['m']
    p = [row[:] for row in kwargs['matrix']]
    memo = {}
    prev_memo = {}
    def get_completion(mask, last):
        key = (mask, last)
        if key in memo:
            return memo[key]
        if mask == (1 << last):
            comp = [p[last][k] for k in range(m)]
            memo[key] = tuple(comp)
            return comp
        min_comp = None
        best_prev = -1
        for prev in range(n):
            if mask & (1 << prev) and prev != last:
                prev_mask = mask ^ (1 << last)
                prev_comp = get_completion(prev_mask, prev)
                comp = [0] * m
                start = prev_comp[0]
                for k in range(m):
                    if k > 0:
                        start = max(prev_comp[k], comp[k-1])
                    comp[k] = start + p[last][k]
                if min_comp is None or comp < min_comp:
                    min_comp = comp
                    best_prev = prev
        if min_comp is not None:
            memo[key] = tuple(min_comp)
            prev_memo[key] = best_prev
            return min_comp
        return None
    full_mask = (1 << n) - 1
    min_makespan = float('inf')
    best_last = -1
    for last in range(n):
        comp = get_completion(full_mask, last)
        makespan = comp[-1]
        if makespan < min_makespan:
            min_makespan = makespan
            best_last = last
    # reconstruct sequence
    seq = []
    current_mask = full_mask
    current_last = best_last
    while current_mask:
        seq.append(current_last + 1)
        key = (current_mask, current_last)
        prev = prev_memo[key]
        current_mask ^= (1 << current_last)
        current_last = prev
    seq.reverse()
    return {'job_sequence': seq}