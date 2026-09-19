# Guidance for the solver below.  This task is a 2D assortment / cutting-stock style
# problem with min/max quantities.  The previous solution is already feasible but
# leaves score on the table by packing only minimum quantities.
additional_context = """
For the assortment problem, deepen the existing shelf-packing solution rather than restarting.

Important improvement target:
- Do NOT pack only the minimum required quantities.
- First build feasible layouts for the minimum counts as before.
- Then exploit optional quantities up to each piece's max bound to fill unused space in already-open stock sheets.
- Prefer adding optional pieces only when they fit into existing bins/shelves, because that lowers the waste fraction without increasing stock count.
- Prioritize optional candidates by larger area and by tight fit into the remaining shelf width/height.
- Try several stock types and choose the feasible solution with the lowest true waste ratio:
    waste = (total stock area used by opened sheets - total placed piece area) / total stock area.
- Keep all coordinates non-overlapping and within stock bounds.  The simple shelf representation from the current solution is good enough; extend it with an optional-fill pass.
- If no mandatory pieces exist, choose optional pieces to create the most filled single stock sheet rather than returning empty or one arbitrary piece.

A helper named fill_optional_shelf_bins is available.  It mutates/extends shelf-style bins by adding optional pieces that fit into existing bins, while respecting max counts.
"""