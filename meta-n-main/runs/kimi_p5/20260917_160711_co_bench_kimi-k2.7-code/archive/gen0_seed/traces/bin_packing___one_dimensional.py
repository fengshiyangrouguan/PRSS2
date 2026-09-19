def solve(**kwargs):
    """
    Solve the one-dimensional bin packing problem for a single test case.
    """
    bin_capacity = int(kwargs.get('bin_capacity', 0))
    items = kwargs.get('items', [])
    n = int(kwargs.get('num_items', len(items)))

    # Pair each item with its 1-based original index.
    indexed_items = [(int(items[i]), i + 1) for i in range(n)]

    # Sort items in decreasing order of size (First-Fit Decreasing).
    indexed_items.sort(key=lambda x: (-x[0], x[1]))

    bins = []      # each bin is a list of (size, 1-based index)
    residual = []  # remaining capacity in each bin

    for size, idx in indexed_items:
        placed = False
        # First-fit placement.
        for b, free in enumerate(residual):
            if free >= size:
                bins[b].append((size, idx))
                residual[b] = free - size
                placed = True
                break
        if not placed:
            bins.append([(size, idx)])
            residual.append(bin_capacity - size)

    # Local improvement: try to eliminate a bin by moving all of its items
    # into other bins that still have enough residual capacity.
    changed = True
    while changed:
        changed = False
        if len(bins) <= 1:
            break

        # Consider the bins in order of increasing total load.
        loads = [sum(size for size, _ in bin_items) for bin_items in bins]
        order = sorted(range(len(bins)), key=lambda i: loads[i])

        for i in order:
            if len(bins) <= 1:
                break

            source_items = bins[i][:]
            # Try to place larger items first; use best-fit destinations.
            source_items.sort(key=lambda x: -x[0])

            temp_free = residual[:]  # simulate moves
            destinations = []
            ok = True

            for size, idx in source_items:
                best_bin = None
                for b, free in enumerate(temp_free):
                    if b == i:
                        continue
                    if free >= size:
                        if best_bin is None or free < temp_free[best_bin]:
                            best_bin = b
                if best_bin is None:
                    ok = False
                    break
                temp_free[best_bin] -= size
                destinations.append(best_bin)

            if ok:
                # Perform the moves.
                for (size, idx), b in zip(source_items, destinations):
                    bins[b].append((size, idx))
                    residual[b] -= size
                # Remove the emptied bin.
                del bins[i]
                del residual[i]
                changed = True
                break

    result_bins = [[idx for _, idx in bin_items] for bin_items in bins]
    return {
        'num_bins': len(result_bins),
        'bins': result_bins
    }