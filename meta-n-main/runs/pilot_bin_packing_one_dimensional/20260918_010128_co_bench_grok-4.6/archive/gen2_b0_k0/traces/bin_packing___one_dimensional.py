def solve(**kwargs):
    id = kwargs.get('id')
    bin_capacity = kwargs.get('bin_capacity')
    items = kwargs.get('items')
    if not items:
        return {'num_bins': 0, 'bins': []}
    n = len(items)
    # Sort indices by item size descending
    sorted_indices = sorted(range(n), key=lambda i: items[i], reverse=True)
    # First Fit Decreasing assignment
    bins = []
    for idx in sorted_indices:
        placed = False
        for bin_idx in range(len(bins)):
            if sum(items[i] for i in bins[bin_idx]) + items[idx] <= bin_capacity:
                bins[bin_idx].append(idx)
                placed = True
                break
        if not placed:
            bins.append([idx])
    num_bins = len(bins)
    # Convert to 1-based indices
    bins_1based = [[i + 1 for i in bin] for bin in bins]
    # Simple local search improvement: limited swaps
    improved = True
    while improved:
        improved = False
        for i in range(num_bins):
            for j in range(i + 1, num_bins):
                for a in bins[i]:
                    for b in bins[j]:
                        if items[a] + items[b] <= bin_capacity:
                            # Swap
                            bins[i].remove(a)
                            bins[j].remove(b)
                            bins[i].append(b)
                            bins[j].append(a)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            num_bins = len(bins)
    bins_1based = [[i + 1 for i in bin] for bin in bins]
    return {
        'num_bins': num_bins,
        'bins': bins_1based
    }