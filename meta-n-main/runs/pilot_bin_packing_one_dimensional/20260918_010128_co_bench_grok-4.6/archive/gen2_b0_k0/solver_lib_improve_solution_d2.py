def improve_solution(bins: list, bin_capacity: int, items: list) -> dict:
    """Refine a bin assignment via 2-item swaps to reduce bins used or waste.

    Args:
        bins: list of lists where each inner list contains 0-based item indices already assigned to a bin.
        bin_capacity: int — maximum item weight per bin.
        items: list[int] — list of item sizes, indexed 0..N-1.

    Returns:
        dict with updated 'num_bins' and 'bins' (0-based inner lists of indices). Only stdlib used; O(N^2) worst-case.
    """
    import copy
    N = len(items)
    best_solution = copy.deepcopy(bins)
    best_num_bins = len(best_solution)

    def compute_remaining(soln):
        rems = [bin_capacity] * len(soln)
        for b_idx, b in enumerate(soln):
            total = sum(items[i] for i in b)
            rems[b_idx] = bin_capacity - total
        return rems

    changed = True
    while changed:
        changed = False
        rems = compute_remaining(best_solution)
        for i in range(len(best_solution)):
            for j in range(i + 1, len(best_solution)):
                if len(best_solution[i]) > 0 and len(best_solution[j]) > 0:
                    item_a = best_solution[i][0]
                    item_b = best_solution[j][0]
                    new_rems = rems[:]
                    new_rems[i] += items[item_b]
                    new_rems[j] += items[item_a]
                    if (new_rems[i] <= bin_capacity and new_rems[j] <= bin_capacity):
                        swapped_bins = copy.deepcopy(best_solution)
                        swapped_bins[i] = [item_a, item_b] + best_solution[i][1:]
                        swapped_bins[j] = best_solution[j]
                        swapped_bins[i].remove(item_b)
                        swapped_bins[j].remove(item_a)
                        swapped_num_bins = len(swapped_bins)
                        if swapped_num_bins < best_num_bins:
                            best_solution = swapped_bins
                            best_num_bins = swapped_num_bins
                            changed = True
                            break
            if changed:
                break
    return {
        'num_bins': best_num_bins,
        'bins': [[i] for b in best_solution for i in b]  # flatten to 0-based indices
    }