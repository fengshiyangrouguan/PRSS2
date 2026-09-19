if "flow_shop_scheduling" in task.description.lower():
    additional_context = "Use the injected makespan helper. Handle n=0 and m=0 gracefully. Sort jobs by total processing time. Start with first 2 jobs then insert remaining ones in best position via makespan. Return 1-based job sequence. Avoid index errors on empty sequences."
else:
    additional_context = ""