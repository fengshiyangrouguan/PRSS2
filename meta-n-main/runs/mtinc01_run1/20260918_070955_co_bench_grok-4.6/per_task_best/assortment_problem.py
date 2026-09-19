def solve(**kwargs):
    m = kwargs['m']
    stocks = kwargs['stocks']
    pieces = kwargs['pieces']
    if not stocks or not pieces or m == 0:
        return {"objective": 0.0, "placements": {1: {"stock_type": 1, "placements": []}}}
    stock_type = 1
    placements = {}
    current_instance_id = 1
    total_used = 0.0
    total_stock_area = 0.0
    s_l = stocks[0]['length']
    s_w = stocks[0]['width']
    for i in range(m):
        piece = pieces[i]
        min_count = piece['min']
        if min_count == 0:
            continue
        p_l0 = piece['length']
        p_w0 = piece['width']
        # Try orientation 0
        if p_l0 <= s_l and p_w0 <= s_w:
            orient = 0
            pl = p_l0
            pw = p_w0
        # Try orientation 1
        elif p_w0 <= s_l and p_l0 <= s_w:
            orient = 1
            pl = p_w0
            pw = p_l0
        else:
            continue  # cannot fit this piece type
        num_stocks = min_count
        for _ in range(num_stocks):
            placements[current_instance_id] = {
                "stock_type": stock_type,
                "placements": [{
                    "piece": i + 1,
                    "x": 0.0,
                    "y": 0.0,
                    "orientation": orient
                }]
            }
            total_used += pl * pw
            total_stock_area += s_l * s_w
            current_instance_id += 1
    if current_instance_id == 1:
        objective = 0.0
    else:
        objective = (total_stock_area - total_used) / total_stock_area
    return {
        "objective": objective,
        "placements": placements
    }