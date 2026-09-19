def solve(**kwargs):
    jobs = kwargs.get('jobs', [])
    h = kwargs.get('h', 0.6)
    if not jobs:
        return {'schedule': []}
    # Sort jobs by non-decreasing a/p ratio
    sorted_jobs = sorted(enumerate(jobs), key=lambda x: x[1][1] / x[1][0])
    schedule = [idx + 1 for idx, _ in sorted_jobs]
    return {'schedule': schedule}