if n > 100:
    additional_context = (
        "For large instances (n>100) prefer the fast O(n m) heuristic: "
        "sort jobs by increasing p[0] (first-machine processing time). "
        "Only fall back to the two-job NEH insertion loop when n<=100 or m==2."
    )
else:
    additional_context = ""