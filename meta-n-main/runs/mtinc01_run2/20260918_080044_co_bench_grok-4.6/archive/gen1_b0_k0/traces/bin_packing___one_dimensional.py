def solve(**kwargs):
    bin_capacity = kwargs['bin_capacity']
    items = kwargs['items']
    n = len(items)
    sorted_items = sorted(enumerate(items), key=lambda x: x[1], reverse=True)
    bins = []  # list of (remaining_capacity, list_of_1based_indices)
    for item_idx, size in sorted_items:
        if size > bin_capacity:
            return {'num_bins': 1000000, 'bins': []}
        best_bin = -1
        best_space = bin_capacity + 1
        for b in range(len(bins)):
            rem = bins[b][0]
            if rem >= size and rem - size < best_space:
                best_space = rem - size
                best_bin = b
        if best_bin != -1:
            bins[best_bin][0] -= size
            bins[best_bin][1].append(item_idx + 1)
        else:
            bins.append([bin_capacity - size, [item_idx + 1]])
    num_bins = len(bins)
    return {'num_bins': num_bins, 'bins': [b[1] for b in bins]}