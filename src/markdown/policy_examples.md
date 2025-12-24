# Policy Examples from the Eudoxia paper

## A naive scheduler

This is a simple FIFO scheduler that assigns one job at a time to each pool created. It allocates all available resources of a pool to a single pipeline.

```python
@register_scheduler_init(key="naive")
def naive_pipeline_init(s):
    s.waiting_queue = []
    s.multi_operator_containers = s.params["multi_operator_containers"]

@register_scheduler(key="naive")
def naive_pipeline(s, results, pipelines):
    """
    A naive implementation of a scheduling pipeline which allocates all
    resources of a pool to a single pipeline and handles them in a FIFO order. It
    assigns one job at a time to each pool created.
    Args:
        results: List of ExecutionResult from previous tick (ignored in naive implementation)
        pipelines: List of pipelines from WorkloadGenerator
    Returns:
        Tuple of (suspensions, assignments):
            - suspensions: List of containers to suspend (always empty for naive scheduler)
            - assignments: List of new assignments to provide to Executor
    """
    # Exit fast if nothing new
    if len(pipelines) == 0 and len(results) == 0:
        return [], []

    for p in pipelines:
        s.waiting_queue.append(p)

    suspensions = []
    assignments = []
    requeue_pipelines = []

    for pool_id in range(s.executor.num_pools):
        avail_cpu_pool = s.executor.pools[pool_id].avail_cpu_pool
        avail_ram_pool = s.executor.pools[pool_id].avail_ram_pool
        if avail_cpu_pool <= 0 or avail_ram_pool <= 0:
            continue

        while s.waiting_queue:
            pipeline = s.waiting_queue.pop(0)
            if pipeline.runtime_status().is_pipeline_successful():
                continue  # Already done

            requeue_pipelines.append(pipeline)

            # Get operators ready to run
            if s.multi_operator_containers:
                op_list = pipeline.runtime_status().get_ops(ASSIGNABLE_STATES, require_parents_complete=False)
            else:
                op_list = pipeline.runtime_status().get_ops(ASSIGNABLE_STATES, require_parents_complete=True)[:1]
            if not op_list:
                continue

            assignment = Assignment(ops=op_list, cpu=avail_cpu_pool, ram=avail_ram_pool,
                                    priority=pipeline.priority, pool_id=pool_id,
                                    pipeline_id=pipeline.pipeline_id)
            assignments.append(assignment)
            break

    s.waiting_queue.extend(requeue_pipelines)
    return suspensions, assignments
```

## A priority scheduler

This scheduler assigns jobs to resources purely in order of priority. This function WILL STARVE low priority jobs.

```python
@register_scheduler_init(key="priority")
def init_priority_scheduler(s):
    s.multi_operator_containers = s.params["multi_operator_containers"]
    s.qry_jobs = []      # high priority
    s.interactive_jobs = []  # medium priority
    s.batch_ppln_jobs = []   # low priority
    s.queues_by_prio = {
        Priority.QUERY: s.qry_jobs,
        Priority.INTERACTIVE: s.interactive_jobs,
        Priority.BATCH_PIPELINE: s.batch_ppln_jobs,
    }
    s.suspending = {}
    s.oom_failed_to_run = 0

def get_pool_with_max_avail_ram(s, pool_stats):
    id_ = -1
    max_ram = 0
    for i in range(s.executor.num_pools):
        if (pool_stats[i]["avail_cpu"] > 0) and (max_ram < pool_stats[i]["avail_ram"]):
            id_ = i
            max_ram = pool_stats[i]["avail_ram"]
    return id_

@register_scheduler(key="priority")
def priority_scheduler(s, results, pipelines):
    """
    Assign jobs to resources purely in order of priority.
    This function WILL STARVE low priority jobs.

    Args:
        results: List of ExecutionResult from executor last tick (successes and failures)
        pipelines: Newly generated pipelines arriving
    Returns:
        Tuple of (suspensions, assignments):
            - suspensions: List of containers to suspend (for preemption)
            - assignments: List of new assignments to provide to Executor
    """
    # Collect all pipelines that need processing
    pipelines_to_process = {p.pipeline_id: p for p in pipelines}
    for r in results:
        for op in r.ops:
            pipelines_to_process[op.pipeline.pipeline_id] = op.pipeline

    # Build retry info from failed results
    retry_info = {}
    for r in results:
        if r.failed():
            for op in r.ops:
                if op.state() != OperatorState.COMPLETED:
                    retry_info[op.id] = {
                        "old_ram": r.ram,
                        "old_cpu": r.cpu,
                        "error": r.error,
                        "container_id": r.container_id,
                        "pool_id": r.pool_id,
                    }

    # Queue new work
    if pipelines_to_process:
        already_queued = set()
        for queue in s.queues_by_prio.values():
            for job in queue:
                for op in job["ops"]:
                    already_queued.add(op.id)

        for pipeline in pipelines_to_process.values():
            if s.multi_operator_containers:
                op_list = pipeline.runtime_status().get_ops(ASSIGNABLE_STATES, require_parents_complete=False)
            else:
                op_list = pipeline.runtime_status().get_ops(ASSIGNABLE_STATES, require_parents_complete=True)
            op_list = [op for op in op_list if op.id not in already_queued]
            if len(op_list) == 0:
                continue

            job = {"priority": pipeline.priority, "pipeline": pipeline, "ops": op_list,
                   "retry_stats": retry_info.get(op_list[0].id)}
            s.queues_by_prio[pipeline.priority].append(job)

    # Resource stats per pool
    pool_stats = {}
    for i in range(s.executor.num_pools):
        pool_stats[i] = {
            "avail_cpu": s.executor.pools[i].avail_cpu_pool,
            "avail_ram": s.executor.pools[i].avail_ram_pool,
            "total_cpu": s.executor.pools[i].max_cpu_pool,
            "total_ram": s.executor.pools[i].max_ram_pool,
        }

    # Process queues in priority order
    queues = [s.qry_jobs, s.interactive_jobs, s.batch_ppln_jobs]
    new_assignments = []

    for queue in queues:
        to_remove = []
        to_start = []
        for job in queue:
            pool_id = get_pool_with_max_avail_ram(s, pool_stats)
            if pool_id == -1:
                break

            to_remove.append(job)
            op_list = job["ops"]
            rs = job.get("retry_stats")

            # Handle OOM retry with doubled resources
            if rs is not None and rs.get("error") is not None:
                job_cpu = 2 * rs["old_cpu"]
                job_ram = 2 * rs["old_ram"]
                if job_cpu > pool_stats[pool_id]["avail_cpu"] or job_ram > pool_stats[pool_id]["avail_ram"]:
                    continue
                cpu_ratio = job_cpu / pool_stats[pool_id]["total_cpu"]
                ram_ratio = job_ram / pool_stats[pool_id]["total_ram"]
                if cpu_ratio >= 0.5 or ram_ratio >= 0.5:
                    s.oom_failed_to_run += 1
                    continue
            elif rs is not None:
                job_cpu = rs["old_cpu"]
                job_ram = rs["old_ram"]
            else:
                job_cpu = max(1, int(pool_stats[pool_id]["total_cpu"] / 10))
                job_ram = max(1, int(pool_stats[pool_id]["total_ram"] / 10))
                if job_cpu >= pool_stats[pool_id]["avail_cpu"] or job_ram >= pool_stats[pool_id]["avail_ram"]:
                    job_cpu = pool_stats[pool_id]["avail_cpu"]
                    job_ram = pool_stats[pool_id]["avail_ram"]

            asgmnt = Assignment(ops=op_list, cpu=job_cpu, ram=job_ram,
                                priority=job["priority"], pool_id=pool_id,
                                pipeline_id=job["pipeline"].pipeline_id if job["pipeline"] else "unknown")
            pool_stats[pool_id]["avail_cpu"] -= job_cpu
            pool_stats[pool_id]["avail_ram"] -= job_ram
            to_start.append(asgmnt)

        new_assignments.extend(to_start)
        for j in to_remove:
            queue.remove(j)

    # Handle preemption for high priority jobs
    suspensions = []
    if len(s.qry_jobs) > 0:
        num_to_suspend = len(s.qry_jobs)
        cnt = 0
        to_suspend = []
        pool_id = 0
        exhausted = [False for _ in range(s.executor.num_pools)]
        iters = [iter(s.executor.pools[i].active_containers) for i in range(s.executor.num_pools)]

        while cnt < num_to_suspend:
            if all(exhausted):
                break
            try:
                container = next(iters[pool_id])
                while container.priority == Priority.QUERY:
                    container = next(iters[pool_id])
                if container.can_suspend_container():
                    cnt += 1
                    to_suspend.append(container)
            except StopIteration:
                exhausted[pool_id] = True
            pool_id = ((pool_id + 1) % s.executor.num_pools)

        for sus in to_suspend:
            suspensions.append(Suspend(sus.container_id, sus.pool_id))

    return suspensions, new_assignments
```

## A priority-by-pool scheduler

This scheduler assigns jobs to resources based on their priority and the available resources in each pool, i.e. interactive/query jobs go to one pool, batch jobs to another.

```python
@register_scheduler_init(key="priority-pool")
def init_priority_pool_scheduler(s):
    s.multi_operator_containers = s.params["multi_operator_containers"]
    s.qry_jobs = []      # high priority
    s.interactive_jobs = []  # medium priority
    s.batch_ppln_jobs = []   # low priority
    s.queues_by_prio = {
        Priority.QUERY: s.qry_jobs,
        Priority.INTERACTIVE: s.interactive_jobs,
        Priority.BATCH_PIPELINE: s.batch_ppln_jobs,
    }
    s.suspending = {}
    assert s.executor.num_pools == 2, "Priority-by-pool scheduler requires 2 pools"
    s.oom_failed_to_run = 0

@register_scheduler(key="priority-pool")
def priority_pool_scheduler(s, results, pipelines):
    """
    Scheduler that assigns jobs to dedicated pools based on priority.
    Interactive/query jobs go to pool 0, batch jobs to pool 1.
    This scheduler doesn't perform preemption/suspension.

    Args:
        results: List of ExecutionResult from executor last tick
        pipelines: Newly generated pipelines arriving
    Returns:
        Tuple of (suspensions, assignments):
            - suspensions: List of containers to suspend (always empty for this scheduler)
            - assignments: List of new assignments to provide to Executor
    """
    # Collect all pipelines that need processing
    pipelines_to_process = {p.pipeline_id: p for p in pipelines}
    for r in results:
        for op in r.ops:
            pipelines_to_process[op.pipeline.pipeline_id] = op.pipeline

    # Build retry info from failed results
    retry_info = {}
    for r in results:
        if r.failed():
            for op in r.ops:
                if op.state() != OperatorState.COMPLETED:
                    retry_info[op.id] = {
                        "old_ram": r.ram,
                        "old_cpu": r.cpu,
                        "error": r.error,
                        "container_id": r.container_id,
                        "pool_id": r.pool_id,
                    }

    # Queue new work
    if pipelines_to_process:
        already_queued = set()
        for queue in s.queues_by_prio.values():
            for job in queue:
                for op in job["ops"]:
                    already_queued.add(op.id)

        for pipeline in pipelines_to_process.values():
            if s.multi_operator_containers:
                op_list = pipeline.runtime_status().get_ops(ASSIGNABLE_STATES, require_parents_complete=False)
            else:
                op_list = pipeline.runtime_status().get_ops(ASSIGNABLE_STATES, require_parents_complete=True)
            op_list = [op for op in op_list if op.id not in already_queued]
            if len(op_list) == 0:
                continue

            job = {"priority": pipeline.priority, "pipeline": pipeline, "ops": op_list,
                   "retry_stats": retry_info.get(op_list[0].id)}
            s.queues_by_prio[pipeline.priority].append(job)

    # Resource stats per pool
    pool_stats = {}
    for i in range(s.executor.num_pools):
        pool_stats[i] = {
            "avail_cpu": s.executor.pools[i].avail_cpu_pool,
            "avail_ram": s.executor.pools[i].avail_ram_pool,
            "total_cpu": s.executor.pools[i].max_cpu_pool,
            "total_ram": s.executor.pools[i].max_ram_pool,
        }

    # Pool 0: query + interactive, Pool 1: batch
    pool_queues = {0: [s.qry_jobs, s.interactive_jobs],
                   1: [s.batch_ppln_jobs]}
    new_assignments = []

    for pool_id in range(s.executor.num_pools):
        queues = pool_queues[pool_id]
        for queue in queues:
            to_remove = []
            to_start = []
            for job in queue:
                avail_ram = pool_stats[pool_id]["avail_ram"]
                avail_cpu = pool_stats[pool_id]["avail_cpu"]

                if avail_ram == 0 or avail_cpu == 0:
                    break

                to_remove.append(job)
                op_list = job["ops"]
                rs = job.get("retry_stats")

                # Handle OOM retry
                if rs is not None and rs.get("error") is not None:
                    job_cpu = 2 * rs["old_cpu"]
                    job_ram = 2 * rs["old_ram"]
                    cpu_ratio = job_cpu / pool_stats[pool_id]["total_cpu"]
                    ram_ratio = job_ram / pool_stats[pool_id]["total_ram"]
                    if cpu_ratio >= 0.5 or ram_ratio >= 0.5:
                        s.oom_failed_to_run += 1
                        continue
                    if job_cpu >= avail_cpu or job_ram >= avail_ram:
                        job_cpu = avail_cpu
                        job_ram = avail_ram
                elif rs is not None and (rs["old_cpu"] <= avail_cpu and rs["old_ram"] <= avail_ram):
                    job_cpu = rs["old_cpu"]
                    job_ram = rs["old_ram"]
                else:
                    job_cpu = max(1, int(pool_stats[pool_id]["total_cpu"] / 10))
                    job_ram = max(1, int(pool_stats[pool_id]["total_ram"] / 10))
                    if job_cpu >= avail_cpu or job_ram >= avail_ram:
                        job_cpu = avail_cpu
                        job_ram = avail_ram

                asgmnt = Assignment(ops=op_list, cpu=job_cpu, ram=job_ram,
                                    priority=job["priority"], pool_id=pool_id,
                                    pipeline_id=job["pipeline"].pipeline_id if job["pipeline"] else "unknown")
                pool_stats[pool_id]["avail_cpu"] -= job_cpu
                pool_stats[pool_id]["avail_ram"] -= job_ram
                to_start.append(asgmnt)

            new_assignments.extend(to_start)
            for j in to_remove:
                queue.remove(j)

    return [], new_assignments  # No suspensions for this scheduler
```