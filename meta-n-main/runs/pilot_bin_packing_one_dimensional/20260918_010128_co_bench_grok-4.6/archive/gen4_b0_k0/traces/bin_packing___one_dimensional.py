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
    id = kwargs.get('id')
    bin_capacity = kwargs.get('bin_capacity')
    items = kwargs.get('items', [])
    if not items:
        return {'num_bins': 0, 'bins': []}
    n = len(items)
    # Sort indices by item size descending (FFD heuristic)
    indices = sorted(range(n), key=lambda i: items[i], reverse=True)
    bins = []
    for idx in indices:
        placed = False
        for bin_list in bins:
            if sum(items[i] for i in bin_list) + items[idx] <= bin_capacity:
                bin_list.append(idx)
                placed = True
                break
        if not placed:
            bins.append([idx])
    num_bins = len(bins)
    # Convert to 1-based indices
    bins_1based = [[i + 1 for i in bin] for bin in bins]
    return {
        'num_bins': num_bins,
        'bins': bins_1based
    }