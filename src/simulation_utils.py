from functools import wraps
import os
from time import time
from typing import List

# eudoxia imports
from eudoxia.simulator import run_simulator
from eudoxia.workload import WorkloadGenerator
from eudoxia.workload.csv_io import (
    CSVWorkloadWriter,
    WorkloadTraceGenerator,
    CSVWorkloadReader,
)


test_policy_as_string = """
@register_scheduler_init(key="policy2")
def naive_pipeline_init(s):
    s.waiting_queue: Tuple[List[Operator], Priority] = []

@register_scheduler(key="policy2")
def naive_pipeline(s, failures: List[Failure],
                   pipelines: List[Pipeline]) -> Tuple[List[Suspend], List[Assignment]]:
    for p in pipelines:
        for op in p.values:
            s.waiting_queue.append(([op], p.priority))
    if len(s.waiting_queue) == 0:
        return [], []

    suspensions = []
    assignments = []
    for pool_id in range(s.executor.num_pools):
        avail_cpu_pool = s.executor.pools[pool_id].avail_cpu_pool
        avail_ram_pool = s.executor.pools[pool_id].avail_ram_pool
        if avail_cpu_pool > 0 and avail_ram_pool > 0 and s.waiting_queue:
            op_list, priority = s.waiting_queue.pop(0)
            assignment = Assignment(ops=op_list, cpu=avail_cpu_pool, ram=avail_ram_pool,
                                    priority=priority, pool_id=pool_id)
            assignments.append(assignment)
    return suspensions, assignments
""".strip()


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        if "TIMING" in os.environ:
            print(
                "func:%r args:[%r, %r] took: %2.4f sec"
                % (f.__name__, args, kw, te - ts)
            )
        return result

    return wrap


@timing
def generate_trace_file(params, output_file):
    """Generate a trace file with the given parameters"""
    # Generate workload
    workload = WorkloadGenerator(**params)

    # Create trace generator
    trace_generator = WorkloadTraceGenerator(
        workload=workload,
        ticks_per_second=params["ticks_per_second"],
        duration_secs=params["duration"],
    )

    # Write trace to CSV
    with open(output_file, "w") as f:
        writer = CSVWorkloadWriter(f)
        for row in trace_generator.generate_rows():
            writer.write_row(row)


@timing
def run_simulation_with_trace(params, trace_file):
    """Run simulation using a trace file"""
    with open(trace_file) as f:
        reader = CSVWorkloadReader(f)
        workload = reader.get_workload(params["ticks_per_second"])
        return run_simulator(params, workload=workload)


@timing
def generate_traces(k: int, base_params: dict):
    """Generate a bunch of traces with different parameters for length, concurrency etc."""
    traces = []
    for i in range(k):
        trace_file_name = f"trace_{i}_{base_params['duration']}s.csv"
        params = base_params.copy()
        # TODO: adjust whatever params we want, other than just seed
        params["random_seed"] = 10 + i
        generate_trace_file(params, trace_file_name)
        traces.append(trace_file_name)

    return traces


@timing
def get_stats_for_policy(
    # basic simulation params
    base_params: dict,
    # files with traces to use as sim input
    trace_files: list,
    # policy to tests, list of strings, string
    # should be a policy key defined earlier
    policy_algorithm: str,
    # metric to return: "latency" or "throughput"
    metric: str = "throughput",
) -> List[float]:
    """Get raw stats for a given policy, without combining them into a reward

    Args:
        base_params: Basic simulation parameters
        trace_files: List of trace files to use as simulation input
        policy_algorithm: Policy key to test
        metric: Metric to return - either "latency" or "throughput"

    Returns:
        List of metric values (one per trace file)
    """
    latencies = []
    throughput = []
    params = base_params.copy()
    params["scheduler_algo"] = policy_algorithm
    for t in trace_files:
        output = run_simulation_with_trace(params, t)
        latencies.append(output.p99_latency)
        throughput.append(output.throughput)

    if metric == "latency":
        return latencies
    else:  # default to throughput
        return throughput
