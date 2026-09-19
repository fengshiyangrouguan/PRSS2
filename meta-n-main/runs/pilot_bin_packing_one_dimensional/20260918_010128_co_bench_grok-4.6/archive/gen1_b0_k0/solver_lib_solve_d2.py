def solve(**kwargs):
    """Pack items into bins using First Fit Decreasing heuristic.

    Args:
        bin_capacity: int — maximum capacity per bin.
        items: list[float] — list of item sizes (positive numbers).

    Returns:
        dict — {'num_bins': int, 'bins': list[list[int]]} where each inner list is 1-based item indices assigned to that bin.
    """
    bin_capacity = kwargs['bin_capacity']
    items = kwargs['items']
    N = len(items)
    items_with_idx = [(items[i], i) for i in range(N)]
    items_with_idx.sort(key=lambda x: x[0], reverse=True)
    bins = []
    for size, idx in items_with_idx:
        placed = False
        for b in bins:
            rem = bin_capacity - sum(items[j] for j in b)
            if rem >= size:
                b.append(idx)
                placed = True
                break
        if not placed:
            bins.append([idx])
    num_bins = len(bins)
    bins = [[i + 1 for i in b] for b in bins]
    return {
        'num_bins': num_bins,
        'bins': bins
    }