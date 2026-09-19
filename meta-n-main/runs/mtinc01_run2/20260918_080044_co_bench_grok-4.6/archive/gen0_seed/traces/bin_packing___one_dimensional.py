def solve(**kwargs):
    bin_capacity = kwargs['bin_capacity']
    items = kwargs['items']
    n = len(items)
    indexed_items = sorted(enumerate(items), key=lambda x: x[1], reverse=True)
    bins = []
    for i in range(n):
        item_idx, size = indexed_items[i]
        placed = False
        for b in bins:
            if sum(items[j] for j in b) + size <= bin_capacity:
                b.append(item_idx)
                placed = True
                break
        if not placed:
            bins.append([item_idx])
    num_bins = len(bins)
    bins_1based = [[idx + 1 for idx in b] for b in bins]
    return {
        'num_bins': num_bins,
        'bins': bins_1based
    }