def solve(**kwargs):
    bin_capacity = kwargs['bin_capacity']
    items = kwargs['items']
    N = len(items)
    items_with_idx = [(items[i], i) for i in range(N)]
    items_with_idx.sort(key=lambda x: x[0], reverse=True)
    bins = []
    for size, idx in items_with_idx:
        placed = False
        best_bin = None
        best_rem = bin_capacity + 1
        for i, b in enumerate(bins):
            rem = bin_capacity - sum(items[j] for j in b)
            if rem >= size and rem < best_rem:
                best_rem = rem
                best_bin = i
        if best_bin is not None:
            bins[best_bin].append(idx)
            placed = True
        if not placed:
            bins.append([idx])
    num_bins = len(bins)
    bins = [[i+1 for i in b] for b in bins]
    return {
        'num_bins': num_bins,
        'bins': bins
    }