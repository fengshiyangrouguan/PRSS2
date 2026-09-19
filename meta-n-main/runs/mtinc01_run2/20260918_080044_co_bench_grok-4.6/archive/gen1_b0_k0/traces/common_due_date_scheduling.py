def solve(**kwargs):
    """
    Solves the restricted single‐machine common due date scheduling problem.

    The problem:
       Given a list of jobs where each job is represented as a tuple (p, a, b):
         • p: processing time
         • a: earliness penalty coefficient
         • b: tardiness penalty coefficient
       and an optional parameter h (default 0.6), the common due date is computed as:
             d = floor(sum(p) * h)
       A schedule (i.e., a permutation of job indices in 1‐based numbering) is produced.
       When processing the jobs in that order, the penalty is computed by:
         • Adding a × (d − C) if a job’s completion time C is less than d,
         • Adding b × (C − d) if C is greater than d,
         • No penalty if C equals d.
       The objective is to minimize the total penalty.

    Input kwargs:
         - 'jobs' (List[Tuple[int, int, int]]): a list of tuples where each tuple represents a job with:
              • p (int): processing time,
              • a (int): earliness penalty coefficient,
              • b (int): tardiness penalty coefficient.
         - Optional: 'h' (float): the factor used to compute the common due date (default is 0.6).

    Evaluation Metric:
         The computed schedule is evaluated by accumulating processing times and applying
         the appropriate earliness/tardiness penalties with respect to the common due date.

    Returns:
         A dictionary with key 'schedule' whose value is a list of integers representing
         a valid permutation of job indices (1-based).
    """
    jobs = kwargs.get('jobs', [])
    if not jobs:
        return {'schedule': []}
    n = len(jobs)
    # Sort indices in non-decreasing order of a_i / b_i (classic optimal rule for E/T with common due date)
    indices = list(range(n))
    indices.sort(key=lambda i: jobs[i][1] / jobs[i][2] if jobs[i][2] != 0 else float('inf'))
    return {'schedule': [i + 1 for i in indices]}