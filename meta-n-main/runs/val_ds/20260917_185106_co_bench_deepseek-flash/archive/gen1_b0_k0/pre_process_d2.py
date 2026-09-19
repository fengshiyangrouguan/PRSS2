# Detect a 1D bin-packing task by structure/keywords (no task_id literals).
desc = (getattr(task, 'description', '') or '').lower()
meta = str(getattr(task, 'metadata', '') or '').lower()
blob = desc + " " + meta

signals = [
    'bin pack', 'binpack', 'bin_packing', 'bin packing',
    'bin capacity', 'bin_capacity', 'one-dimensional bin', '1d bin',
    'number of bins', 'num_bins',
]

if any(sig in blob for sig in signals):
    additional_context = (
        "BIN PACKING NOTE: A high-quality 1D bin-packing routine is available in the "
        "library as `solve_bin_packing(bin_capacity, items, time_limit)`.\n"
        "It returns `(num_bins, bins)` where `bins` is a list of lists of **0-based** "
        "item indices that exactly partition `range(len(items))`.\n\n"
        "Use it directly rather than writing your own heuristic:\n"
        "