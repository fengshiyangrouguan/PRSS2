def solve(**kwargs):
    bin_capacity = kwargs['bin_capacity']
    items = kwargs['items']
    n = len(items)
    # Sort items descending by size, keep original indices (0-based)
    indexed_items = sorted(enumerate(items), key=lambda x: x[1], reverse=True)
    # bins: list of current fill levels
    bins = []
    bin_assignments = []  # maps each sorted item to its bin index
    for idx, size in indexed_items:
        placed = False
        for b in range(len(bins)):
            if bins[b] + size <= bin_capacity:
                bins[b] += size
                bin_assignments.append(b)
                placed = True
                break
        if not placed:
            bins.append(size)
            bin_assignments.append(len(bins) - 1)
    num_bins = len(bins)
    # Build result bins: each inner list collects 1-based original indices
    result_bins = [[] for _ in range(num_bins)]
    for i, b in enumerate(bin_assignments):
        result_bins[b].append(i + 1)
    # Sort each bin's indices for deterministic output
    for b in result_bins:
        b.sort()
    return {
        'num_bins': num_bins,
        'bins': result_bins
    }