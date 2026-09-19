def solve(**kwargs):
    m = kwargs.get('m', 0)
    stocks = kwargs.get('stocks', [])
    pieces = kwargs.get('pieces', [])
    if not stocks or not pieces:
        return {
            "objective": 0.0,
            "placements": {}
        }
    try:
        from solver_lib import pack_stocks
        from itertools import combinations
        best_objective = float('inf')
        best_placements = None
        n_stocks = len(stocks)
        for k in range(1, 3):
            for combo in combinations(range(n_stocks), k):
                selected_stocks = [stocks[i] for i in combo]
                placements = pack_stocks(selected_stocks, pieces)
                if placements is None:
                    continue
                total_stock_area = sum(s['length'] * s['width'] for s in selected_stocks)
                total_used_area = 0.0
                for stock_id, stock_data in placements.items():
                    for p in stock_data['placements']:
                        piece_idx = p['piece'] - 1
                        if piece_idx < 0 or piece_idx >= len(pieces):
                            continue
                        px = p['x']
                        py = p['y']
                        orient = p['orientation']
                        pl = pieces[piece_idx]['length']
                        pw = pieces[piece_idx]['width']
                        if orient == 1:
                            pl, pw = pw, pl
                        total_used_area += pl * pw
                objective = (total_stock_area - total_used_area) / total_stock_area if total_stock_area > 0 else 0.0
                if objective < best_objective:
                    best_objective = objective
                    best_placements = placements
        if best_placements is None:
            return {
                "objective": 0.0,
                "placements": {
                    1: {
                        "stock_type": 1,
                        "placements": []
                    }
                }
            }
        return {
            "objective": best_objective,
            "placements": best_placements
        }
    except Exception:
        return {
            "objective": 0.0,
            "placements": {
                1: {
                    "stock_type": 1,
                    "placements": []
                }
            }
        }