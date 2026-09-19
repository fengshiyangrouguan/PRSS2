additional_context = """
Assortment/cutting-stock structural guidance:

The existing shelf-based solver is feasible and should remain a fallback, but its main weakness is shelf fragmentation. For rectangular assortment instances with stock rectangles and piece min/max quantities, call the helper:

    solve_assortment_free_rect(stocks, pieces, m, deadline)

This helper returns a complete solution dictionary using a free-rectangle packing strategy, including optional quantity scenarios and mixed stock-type sets. Compare its returned objective against the current shelf solution and return the lower true waste ratio.

Recommended integration:
1. Build the current shelf solution as before.
2. Also call solve_assortment_free_rect with the same inputs and a deadline slightly before the global timeout.
3. If the helper returns a feasible non-empty solution with a smaller objective, use it.
4. Keep the shelf solution as fallback because shelves can still be better on highly regular banded instances.

Do not key behavior on file names or task_id. Detect this structure by the presence of rectangular stocks, rectangular pieces, and min/max quantity bounds.
"""