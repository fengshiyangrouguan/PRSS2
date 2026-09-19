import itertools
import sys

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
    n = len(jobs)
    if n == 0:
        return {'schedule': []}
    h = kwargs.get('h', 0.6)
    P = sum(p for p, a, b in jobs)
    d = int(P * h)
    best_penalty = sys.maxsize
    best_perm = None
    for perm in itertools.permutations(range(n)):
        C = 0
        penalty = 0
        for idx in perm:
            p, a, b = jobs[idx]
            C += p
            if C < d:
                penalty += a * (d - C)
            elif C > d:
                penalty += b * (C - d)
        if penalty < best_penalty:
            best_penalty = penalty
            best_perm = perm
    return {'schedule': [i + 1 for i in best_perm]}