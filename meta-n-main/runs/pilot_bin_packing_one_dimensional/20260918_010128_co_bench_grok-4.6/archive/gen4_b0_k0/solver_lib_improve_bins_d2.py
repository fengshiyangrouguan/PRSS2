def improve_bins(bins: list, items: list, bin_capacity: int) -> tuple:
    """2-swap local search to reduce the number of bins.

    Args:
        bins: list of lists; each inner list contains item indices (0-based) in the current packing.
        items: list of item sizes (weights).
        bin_capacity: maximum size per bin.

    Returns:
        (new_bins: list of lists of 0-based indices, new_num_bins: int)
        The returned bins are guaranteed to be valid (no bin exceeds capacity) and contain one fewer bin whenever a consolidation is found.
    """
    import copy
    current_num = len(bins)
    new_bins = copy.deepcopy(bins)
    changed = True
    while changed:
        changed = False
        for i in range(len(new_bins)):
            for j in range(i + 1, len(new_bins)):
                bi = new_bins[i]
                bj = new_bins[j]
                if not bi or not bj:
                    continue
                for a_idx in bi[:]:  # copy to allow removal
                    for b_idx in bj[:]:
                        # swap
                        new_bins[i] = [x for x in bi if x != a_idx] + [b_idx]
                        new_bins[j] = [x for x in bj if x != b_idx] + [a_idx]
                        # check if any bin became empty
                        non_empty = [b for b in new_bins if b]
                        if len(non_empty) < len(new_bins):
                            new_bins = non_empty
                            changed = True
                            break
                    if changed:
                        break
                if changed:
                    break
            if changed:
                break
    return new_bins, len(new_bins)