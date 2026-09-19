def solve(**kwargs):
    bin_capacity = kwargs.get('bin_capacity', 0)
    items = kwargs.get('items', [])
    n = len(items)
    if n == 0:
        return {'num_bins': 0, 'bins': []}
    indexed_items = sorted(enumerate(items, 1), key=lambda x: x[1], reverse=True)
    bins = []
    bin_sums = []
    for idx, size in indexed_items:
        placed = False
        for i in range(len(bins)):
            if bin_sums[i] + size <= bin_capacity:
                bins[i].append(idx)
                bin_sums[i] += size
                placed = True
                break
        if not placed:
            bins.append([idx])
            bin_sums.append(size)
    num_bins = len(bins)
    return {'num_bins': num_bins, 'bins': bins}