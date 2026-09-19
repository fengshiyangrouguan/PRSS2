def solve(**kwargs):
    m = kwargs.get('m', 0)
    stocks = kwargs.get('stocks', [])
    pieces = kwargs.get('pieces', [])
    if m == 0 or not stocks:
        return {"objective": 0.0, "placements": {}}
    # Choose stock type with smallest area
    stock_areas = [s['length'] * s['width'] for s in stocks]
    best_stock_idx = stock_areas.index(min(stock_areas))
    # Place each required copy of each piece in its own stock instance (guarantees validity)
    placements = {}
    instance_id = 1
    total_used = 0.0
    stock_length = stocks[best_stock_idx]['length']
    stock_width = stocks[best_stock_idx]['width']
    for p_idx in range(m):
        piece = pieces[p_idx]
        min_c = piece.get('min', 0)
        for c in range(min_c):
            length = piece['length']
            width = piece['width']
            # Try orientations
            if length <= stock_length and width <= stock_width:
                orient = 0
                l, w = length, width
            elif width <= stock_length and length <= stock_width:
                orient = 1
                l, w = width, length
            else:
                continue  # cannot fit this copy; skip to avoid violation
            stock_placements = [{
                'piece': p_idx + 1,
                'x': 0.0,
                'y': 0.0,
                'orientation': orient
            }]
            placements[instance_id] = {
                'stock_type': best_stock_idx + 1,
                'placements': stock_placements
            }
            total_used += l * w
            instance_id += 1
    num_instances = instance_id - 1
    stock_area = stock_length * stock_width
    if stock_area == 0 or num_instances == 0:
        waste = 0.0
    else:
        waste = (num_instances * stock_area - total_used) / (num_instances * stock_area)
    return {
        "objective": waste,
        "placements": placements
    }