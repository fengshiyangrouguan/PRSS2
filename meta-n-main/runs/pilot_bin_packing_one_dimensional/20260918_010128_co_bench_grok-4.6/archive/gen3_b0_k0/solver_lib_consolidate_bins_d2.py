def consolidate_bins(bins: list[list[int]], items: list[int], bin_capacity: int) -> list[list[int]]:
    """Post-process bins to reduce their count by moving items into other bins' free space.

    Args:
        bins: list of lists, each inner list is 0-based item indices in that bin.
        items: list of item sizes.
        bin_capacity: int, maximum size per bin.

    Returns:
        list of lists: updated bins with possibly fewer bins (emptied bins removed).
    """
    import copy
    current_bins = copy.deepcopy(bins)
    changed = True
    while changed:
        changed = False
        for i in range(len(current_bins) - 1, -1, -1):  # try emptying largest bins first
            target_bin = current_bins[i]
            if not target_bin:
                continue
            # compute free spaces of all other bins
            free_spaces = []
            for j, b in enumerate(current_bins):
                if j != i:
                    rem = bin_capacity - sum(items[k] for k in b)
                    free_spaces.append((rem, j))
            free_spaces.sort(reverse=True)  # largest free first for better fit chance
            can_pack = True
            for item_idx in target_bin[:]:
                size = items[item_idx]
                placed = False
                for rem, tgt in free_spaces:
                    if rem >= size:
                        current_bins[tgt].append(item_idx)
                        placed = True
                        break
                if not placed:
                    can_pack = False
                    break
            if can_pack:
                # remove emptied bin
                del current_bins[i]
                changed = True
    return current_bins