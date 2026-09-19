def solve(**kwargs):
    """
    Solve the one-dimensional bin packing problem for a single test case.

    Input kwargs (for a single test case):
      - id:           The problem identifier (string)
      - bin_capacity: The capacity of each bin (int)
      - num_items:    The number of items (int)
      - items:        A list of item sizes (list of ints)
      - **kwargs:     Other unused keyword arguments

    Evaluation metric:
      - The solution is scored by the total number of bins used.
      - If the solution is invalid (e.g., items are missing or duplicated, or bin capacity is exceeded),
        a penalty of 1,000,000 is added.

    Returns:
      A dictionary with:
        - 'num_bins': An integer, the number of bins used.
        - 'bins': A list of lists, where each inner list contains the 1-based indices of items assigned to that bin.

    Note: This is a placeholder implementation.
    """
    bin_capacity = kwargs['bin_capacity']
    items = kwargs['items']
    n = len(items)
    # Create list of (size, 1-based index) and sort by size descending
    indexed_items = [(items[i], i + 1) for i in range(n)]
    indexed_items.sort(reverse=True, key=lambda x: x[0])
    bins = []
    for size, idx in indexed_items:
        placed = False
        for bin_load in bins:
            if sum(bin_load) + size <= bin_capacity:
                bin_load.append(idx)
                placed = True
                break
        if not placed:
            bins.append([idx])
    num_bins = len(bins)
    return {
        'num_bins': num_bins,
        'bins': bins
    }