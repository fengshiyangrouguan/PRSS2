def best_fit_decreasing(items: list, bin_capacity: int) -> tuple:
    """Best-Fit Decreasing heuristic for 1D bin packing.

    Args:
        items: list of item sizes (positive floats <= bin_capacity).
        bin_capacity: max items per bin.

    Returns:
        (num_bins: int, bins: list) where bins is a list of lists each containing item indices in that bin.
    """
    if not items:
        return 0, []
    indexed = sorted(enumerate(items), key=lambda x: x[1], reverse=True)
    bins = []
    for item_idx, size in indexed:
        if size > bin_capacity:
            continue
        best_bin_idx = -1
        best_remaining = bin_capacity + 1
        for b_idx, b in enumerate(bins):
            remaining = bin_capacity - sum(items[j] for j in b)
            if remaining >= size and remaining < best_remaining:
                best_remaining = remaining
                best_bin_idx = b_idx
        if best_bin_idx >= 0:
            bins[best_bin_idx].append(item_idx)
        else:
            bins.append([item_idx])
    return len(bins), [[idx + 1 for idx in b] for b in bins]