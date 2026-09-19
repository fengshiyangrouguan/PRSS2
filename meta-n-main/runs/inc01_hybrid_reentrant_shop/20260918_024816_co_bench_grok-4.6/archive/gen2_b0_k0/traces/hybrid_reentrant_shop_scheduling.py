import heapq

def solve(**kwargs):
    n_jobs = kwargs['n_jobs']
    n_machines = kwargs['n_machines']
    init_time = kwargs['init_time']
    setup_times = kwargs['setup_times']
    processing_times = kwargs['processing_times']
    
    # Simple identity permutation (0-based job indices)
    permutation = list(range(n_jobs))
    
    # Dynamic batch assignment using min-heap on machine loads
    # Load = sum of (init_time + processing_time) for jobs assigned to each machine
    machines = list(range(n_machines))
    load_heap = [(0, m) for m in machines]
    heapq.heapify(load_heap)
    
    batch_assignment = []
    for job in range(n_jobs):
        load, machine = heapq.heappop(load_heap)
        batch_assignment.append(machine)
        new_load = load + init_time + processing_times[job]
        heapq.heappush(load_heap, (new_load, machine))
    
    return {
        'permutation': permutation,
        'batch_assignment': batch_assignment
    }