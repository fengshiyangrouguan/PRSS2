def local_search(schedule, evaluate_fn, time_limit=5.0):
    """One-pass local improvement for job-shop makespan via adjacent swaps on identical machines.

    Args:
        schedule: dict[str, list] — expected keys include 'start_times' (2D list [job][op])
        evaluate_fn: callable — receives a schedule dict and returns a scalar score in [0.0, 1.0]
        time_limit: float — wall-clock seconds to run (default 5.0)

    Returns:
        dict — refined schedule dict with updated 'start_times'
    """
    import time, random
    start = time.time()
    best_schedule = schedule.copy()
    best_score = evaluate_fn(best_schedule)
    changed = True
    while changed and time.time() - start < time_limit:
        changed = False
        mach_op_pairs = {}  # mach -> list of (job, op) tuples on that machine
        for j in range(len(best_schedule["start_times"])):
            for m in range(len(best_schedule["start_times"][j])):
                mach = best_schedule.get("machines", [None] * len(best_schedule["start_times"]))[j][m]
                if mach is not None:
                    if mach not in mach_op_pairs:
                        mach_op_pairs[mach] = []
                    mach_op_pairs[mach].append((j, m))
        for mach, ops in mach_op_pairs.items():
            ops.sort(key=lambda x: (best_schedule["start_times"][x[0]][x[1]], x[1]))
            for i in range(len(ops) - 1):
                j1, m1 = ops[i]
                j2, m2 = ops[i + 1]
                if m1 + 1 != m2:  # only consecutive
                    continue
                if best_schedule.get("machines", [[0]] * len(best_schedule["start_times"]))[j1][m1] != mach or best_schedule.get("machines", [[0]] * len(best_schedule["start_times"]))[j2][m2] != mach:
                    continue
                # simple adjacent swap candidate
                tmp_start = best_schedule["start_times"][j1][m1]
                tmp_pt = best_schedule["times"][j1][m1]
                best_schedule["start_times"][j1][m1] = best_schedule["start_times"][j2][m2]
                best_schedule["start_times"][j2][m2] = tmp_start
                best_schedule["times"][j1][m1] = best_schedule["times"][j2][m2]
                best_schedule["times"][j2][m2] = tmp_pt
                score = evaluate_fn(best_schedule)
                if score > best_score:
                    best_score = score
                    changed = True
                    break
                else:
                    # revert
                    best_schedule["start_times"][j1][m1] = tmp_start
                    best_schedule["start_times"][j2][m2] = tmp_start + tmp_pt - best_schedule["times"][j1][m1]  # crude revert
                    best_schedule["times"][j1][m1] = tmp_pt
                    best_schedule["times"][j2][m2] = tmp_start - tmp_pt
        if not changed and time.time() - start < time_limit / 2:
            break
    return best_schedule