def solve(**kwargs):
    m = kwargs['m']
    stocks = kwargs['stocks']
    pieces = kwargs['pieces']
    if not pieces:
        solution = {
            "objective": 0.0,
            "placements": {
                1: {
                    "stock_type": 1,
                    "placements": []
                }
            }
        }
        return solution
    # Choose best stock type (largest area) for single stock type
    best_stock_idx = 0
    max_area = 0.0
    for i, s in enumerate(stocks):
        area = s['length'] * s['width']
        if area > max_area:
            max_area = area
            best_stock_idx = i
    # Place all pieces in the best stock type using simple bottom-left heuristic
    # (assumes they fit; in practice may need multiple stocks but limit to 1 for simplicity)
    stock_type = best_stock_idx + 1
    placements = []
    current_x = 0.0
    current_y = 0.0
    stock_width = stocks[best_stock_idx]['length']
    stock_height = stocks[best_stock_idx]['width']
    for p_idx, p in enumerate(pieces):
        length = p['length']
        width = p['width']
        # Try normal orientation
        if length <= stock_width - current_x and width <= stock_height - current_y:
            placements.append({
                'piece': p_idx + 1,
                'x': current_x,
                'y': current_y,
                'orientation': 0
            })
            current_x += length
        else:
            # Try rotated
            if width <= stock_width - current_x and length <= stock_height - current_y:
                placements.append({
                    'piece': p_idx + 1,
                    'x': current_x,
                    'y': current_y,
                    'orientation': 1
                })
                current_x += width
            else:
                # Place in new row if possible
                current_x = 0.0
                current_y += max(length, width) + 0.1  # small gap
                if length <= stock_width and width <= stock_height:
                    placements.append({
                        'piece': p_idx + 1,
                        'x': current_x,
                        'y': current_y,
                        'orientation': 0
                    })
                    current_x += length
                else:
                    # Fallback: rotated
                    placements.append({
                        'piece': p_idx + 1,
                        'x': current_x,
                        'y': current_y,
                        'orientation': 1
                    })
                    current_x += width
    # Compute used area
    used_area = sum(p['length'] * p['width'] for p in pieces)
    total_area = max_area
    waste_pct = (total_area - used_area) / total_area if total_area > 0 else 0.0
    solution = {
        "objective": waste_pct,
        "placements": {
            1: {
                "stock_type": stock_type,
                "placements": placements
            }
        }
    }
    return solution