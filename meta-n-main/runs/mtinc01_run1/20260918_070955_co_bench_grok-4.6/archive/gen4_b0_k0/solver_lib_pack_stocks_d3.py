def pack_stocks(stocks: list, pieces: list, min_counts: list, max_counts: list) -> dict:
    """Pack pieces into stocks respecting min/max counts, trying both orientations.

    Args:
        stocks: list of dict with 'length' and 'width' (float)
        pieces: list of dict with 'length', 'width', 'index' (int)
        min_counts: list of int (min per piece)
        max_counts: list of int (max per piece)

    Returns:
        dict with keys 'objective' (float) and 'placements' (dict).
        Each placement value is a dict with 'stock_type' and 'placements' list
        of dicts containing 'piece', 'x', 'y', 'orientation'.
    """
    placements = {}
    current_instance_id = 1
    total_used = 0.0
    total_stock_area = 0.0
    for p_idx, piece in enumerate(pieces):
        pl0 = piece['length']
        pw0 = piece['width']
        min_n = min_counts[p_idx]
        max_n = max_counts[p_idx]
        if min_n == 0:
            continue
        # Try orientation 0
        orient = 0
        pl = pl0
        pw = pw0
        if not (pl <= stocks[0]['length'] and pw <= stocks[0]['width']):
            # Try orientation 1
            orient = 1
            pl = pw0
            pw = pl0
            if not (pl <= stocks[0]['length'] and pw <= stocks[0]['width']):
                continue
        num_stocks = 0
        for _ in range(min_n):
            placements[current_instance_id] = {
                "stock_type": 1,
                "placements": [{
                    "piece": piece['index'] + 1,
                    "x": 0.0,
                    "y": 0.0,
                    "orientation": orient
                }]
            }
            total_used += pl * pw
            total_stock_area += stocks[0]['length'] * stocks[0]['width']
            current_instance_id += 1
        # Place extras up to max if space (simple: one stock for demo)
        extra = max_n - min_n
        for _ in range(extra):
            placements[current_instance_id] = {
                "stock_type": 1,
                "placements": [{
                    "piece": piece['index'] + 1,
                    "x": 0.0,
                    "y": 0.0,
                    "orientation": orient
                }]
            }
            total_used += pl * pw
            total_stock_area += stocks[0]['length'] * stocks[0]['width']
            current_instance_id += 1
    if current_instance_id == 1:
        objective = 0.0
    else:
        objective = (total_stock_area - total_used) / total_stock_area
    return {
        "objective": objective,
        "placements": placements
    }