def solve(**kwargs):
    id = kwargs.get('id', '')
    bin_capacity = kwargs.get('bin_capacity', 0)
    num_items = kwargs.get('num_items', 0)
    items = kwargs.get('items', [])
    if not items or bin_capacity <= 0:
        return {'num_bins': 0, 'bins': []}
    item_list = list(enumerate(items))
    sorted_items = sorted(item_list, key=lambda x: x[1], reverse=True)
    bins = []
    for idx, size in sorted_items:
        placed = False
        for bin_list in bins:
            if sum(items[i] for i in bin_list) + size <= bin_capacity:
                bin_list.append(idx)
                placed = True
                break
        if not placed:
            bins.append([idx])
    num_bins = len(bins)
    # consolidate_bins
    def consolidate_bins(bins, items, bin_capacity):
        n = len(bins)
        changed = True
        while changed:
            changed = False
            for i in range(n - 1, -1, -1):
                bin_i = bins[i]
                if not bin_i:
                    continue
                remaining = bin_capacity - sum(items[j] for j in bin_i)
                for idx in bin_i[:]:
                    size = items[idx]
                    placed = False
                    for j in range(n):
                        if j == i:
                            continue
                        bin_j = bins[j]
                        if sum(items[k] for k in bin_j) + size <= bin_capacity:
                            bin_j.append(idx)
                            placed = True
                            break
                    if placed:
                        bin_i.remove(idx)
                        changed = True
            bins = [b for b in bins if b]
            n = len(bins)
        return bins
    bins = consolidate_bins(bins, items, bin_capacity)
    bins = [[idx + 1 for idx in b] for b in bins]
    return {
        'num_bins': len(bins),
        'bins': bins
    }