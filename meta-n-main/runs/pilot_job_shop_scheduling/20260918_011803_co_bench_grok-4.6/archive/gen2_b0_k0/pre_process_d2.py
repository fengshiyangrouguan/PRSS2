if "job_shop_scheduling" in task.task_id.lower():
    additional_context = (
        "Prioritize using a LPT dispatching rule for better performance. Before scheduling, sort the jobs in descending order of total processing time (sum of all times[j][:] for each job j). Then, iterate over this sorted job order. Within each sorted job, schedule its operations in their original sequence order. For each operation, compute start time as max(prev_finish, machine_avail[m]) and update machine_avail[m] = start + t. This is a simple yet effective list scheduling heuristic that balances load and reduces idle time compared to fixed input order. Use this as your core strategy; no other changes needed."
    )
else:
    additional_context = ""