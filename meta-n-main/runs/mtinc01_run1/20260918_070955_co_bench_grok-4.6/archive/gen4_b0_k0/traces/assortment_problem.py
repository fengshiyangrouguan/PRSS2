def solve(**kwargs):
    """
    Solves the rectangular piece arrangement optimization problem to minimize the overall waste area percentage.

    Given:
      - m (int): Number of piece types.
      - stocks (list of dict): Each dict represents a stock type with keys:
            'length' (float), 'width' (float), 'fixed_cost' (float).
      - pieces (list of dict): Each dict represents a piece type with keys:
            'length' (float), 'width' (float), 'min' (int), 'max' (int), 'value' (float).

    Objective:
      Arrange rectangular pieces (which may be rotated by 90°) into stock rectangles such that the overall waste area percentage is minimized.
      The waste area percentage is computed as:

             Waste Percentage = (Total Stock Area - Total Used Area) / (Total Stock Area)

    Constraints:
      • Each piece must lie entirely within its assigned stock rectangle.
      • Pieces must not overlap within the same stock rectangle.
      • The number of pieces placed for each piece type must lie within its specified minimum and maximum bounds.
      • You may use unlimited many instances of each selected stock type, but the solution can include at most 2 distinct stock types.

    Output:
      Returns a dictionary with two keys (exactly follow this format):
        - "objective": The overall waste area percentage (float) as computed by the evaluation function.
        - "placements": A dictionary mapping stock instance ids (1-indexed) to their placement details.
          Each stock instance is represented by a dictionary with the following keys:
              'stock_type': (the 1-indexed id of the stock type used for this instance),
              'placements': a list of placements for pieces within that stock instance.
                  Each placement is a dict with keys:
                      'piece'       (piece type, 1-indexed, 1 <= piece type <= m),
                      'x'           (x-coordinate of the bottom-left corner),
                      'y'           (y-coordinate of the bottom-left corner),
                      'orientation' (0 for normal, 1 for rotated 90°).

    NOTE: The returned data should adhere to the output format required for evaluation.
    """
    # Dummy solution: Create a single stock instance of the first stock type, with no pieces placed.
    # In a real solution, you would compute placements that respect all constraints and minimize waste.
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