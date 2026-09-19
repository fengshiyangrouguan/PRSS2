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
    """
    bin_capacity = kwargs['bin_capacity']
    items = kwargs['items']
    num_items = len(items)

    # Sort items in descending order to try and fit larger items first
    items.sort(reverse=True)

    # Initialize bins
    bins = []

    for item in items:
        placed = False
        for bin_index, bin in enumerate(bins):
            if sum(items[i] for i in bin) + item <= bin_capacity:
                bin.append(items.index(item) + 1)  # Add 1-based index
                placed = True
                break
        if not placed:
            bins.append([items.index(item) + 1])  # Add new bin with 1-based index

    return {
        'num_bins': len(bins),
        'bins': bins
    }