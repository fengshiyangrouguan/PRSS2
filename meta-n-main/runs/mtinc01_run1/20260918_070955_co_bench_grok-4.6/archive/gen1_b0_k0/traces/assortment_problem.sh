```python
def solve(**kwargs):
    m = kwargs.get('m', 0)
    stocks = kwargs.get('stocks', [])
    pieces = kwargs.get('pieces', [])
    if m == 0 or not pieces:
        return {"objective": 0.0, "placements": {}}
    # Choose up to 2 stock types, prefer those that can fit largest pieces
    stock_areas = [s['length'] * s['width'] for s in stocks]
    stock_indices = sorted(range(len(stocks)), key=lambda i: stock_areas[i], reverse=True)
    used_stock_ids = stock_indices[:2]
    # Parse pieces
    piece_list = []
    for i, p in enumerate(pieces):
        piece_list.append({
            'index': i,
            'length': p['length'],
            'width': p['width'],
            'min': p['min'],
            'max': p['max'],
            'value': p['value']
        })
    # Sort pieces by area descending for better packing
    piece_list.sort(key=lambda p: p['length'] * p['width'], reverse=True)
    # Initialize bins: list of (stock_type_id, placements, cur_w, cur_h) but use simple placement
    bins = []  # list of dicts: {'stock_type': int, 'placements': list, 'w': float, 'h': float}
    placed_count = [0] * m
    for p in piece_list:
        idx = p['index']
        min_n = p['min']
        max_n = p['max']
        pw = p['length']
        ph = p['width']
        placed = 0
        # Try to place min_n pieces of this type
        while placed < min_n:
            placed_in_bin = False
            for b in bins:
                # Simple placement: try at (0,0) if empty bin, or next position
                fit = False
                if not b['placements']:
                    # empty bin, place at 0,0
                    if pw <= b['w'] and ph <= b['h']:
                        b['placements'].append({
                            'piece': idx + 1,
                            'x': 0.0,
                            'y': 0.0,
                            'orientation': 0
                        })
                        placed_in_bin = True
                        placed += 1
                        placed_count[idx] += 1
                        fit = True
                else:
                    # simple: place next to existing, but for demo place at 0,0 if fits
                    if pw <= b['w'] and ph <= b['h']:
                        b['placements'].append({
                            'piece': idx + 1,
                            'x': 0.0,
                            'y': 0.0,
                            'orientation': 0
                        })
                        placed_in_bin = True
                        placed += 1
                        placed_count[idx] += 1
                        fit = True
                if fit:
                    break
            if not placed_in_bin:
                # open new bin with first available stock type
                st_id = used_stock_ids[0]
                new_bin = {
                    'stock_type': st_id + 1,
                    'placements': [],
                    'w': stocks[st_id]['length'],
                    'h': stocks[st_id]['width']
                }
                if pw <= new_bin['w'] and ph <= new_bin['h']:
                    new_bin['placements'].append({
                        'piece': idx + 1,
                        'x': 0.0,
                        'y': 0.0,
                        'orientation': 0
                    })
                    bins.append(new_bin)
                    placed += 1
                    placed_count[idx] += 1
        # Try to place additional up to max if space
        while placed_count[idx] < p['max']:
            placed_in_bin = False
            for b in bins:
                if pw <= b['w'] and ph <= b['h']:
                    b['placements'].append({
                        'piece': idx + 1,
                        'x': 0.0,
                        'y': 0.0,
                        'orientation': 0
                    })
                    placed_count[idx] += 1
                    placed_in_bin = True
                    break
            if not placed_in_bin:
                break
    # Build placements dict
    stock_placements = {}
    for bid, b in enumerate(bins, 1):
        stock_placements[bid] = {
            'stock_type': b['stock_type'],
            'placements': b['placements']
        }
    # Compute objective
    total_stock_area = 0.0
    used_area = 0.0
    for b in bins:
        sa = b['w'] * b['h']
        total_stock_area += sa
        for pl in b['placements']:
            pidx = pl['piece