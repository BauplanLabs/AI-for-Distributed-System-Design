from functools import wraps
import os
from time import time
from typing import List

# eudoxia imports
from eudoxia.simulator import run_simulator, SimulatorStats
from eudoxia.workload import WorkloadGenerator
from eudoxia.workload.csv_io import (
    CSVWorkloadWriter,
    WorkloadTraceGenerator,
    CSVWorkloadReader,
)


test_policy_as_string = """
@register_scheduler_init(key="policy2")
def naive_pipeline_init(s):
    s.waiting_queue = []

@register_scheduler(key="policy2")
def naive_pipeline(s, results, pipelines):
    '''
    A simple test scheduler that queues pipelines and assigns them FIFO.

    Args:
        results: List of ExecutionResult from previous tick (ignored here)
        pipelines: List of newly arrived Pipeline objects
    Returns:
        Tuple of (suspensions, assignments)
    '''
    for p in pipelines:
        s.waiting_queue.append(p)
    if len(s.waiting_queue) == 0:
        return [], []

    suspensions = []
    assignments = []
    for pool_id in range(s.executor.num_pools):
        avail_cpu_pool = s.executor.pools[pool_id].avail_cpu_pool
        avail_ram_pool = s.executor.pools[pool_id].avail_ram_pool
        if avail_cpu_pool > 0 and avail_ram_pool > 0 and s.waiting_queue:
            pipeline = s.waiting_queue.pop(0)
            op_list = [op for op in pipeline.values]
            assignment = Assignment(ops=op_list, cpu=avail_cpu_pool, ram=avail_ram_pool,
                                    priority=pipeline.priority, pool_id=pool_id,
                                    pipeline_id=pipeline.pipeline_id)
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
    """Run simulation using a trace file.

    Returns None if no containers complete (which causes percentile calculation to fail).
    """
    with open(trace_file) as f:
        reader = CSVWorkloadReader(f)
        workload = reader.get_workload(params["ticks_per_second"])
        try:
            return run_simulator(params, workload=workload)
        except IndexError as e:
            # This happens when no containers complete (empty container_tick_times)
            import logging

            logging.getLogger(__name__).warning(
                f"No containers completed in {trace_file}, skipping: {e}"
            )
            return None


@timing
def generate_traces(k: int, base_params: dict, file_name_prefix: str):
    """Generate a bunch of traces with different parameters for length, concurrency etc."""
    traces = []
    for i in range(k):
        trace_file_name = f"{file_name_prefix}_{i}.csv"
        params = base_params.copy()
        # TODO: adjust whatever params we want, other than just seed
        params["random_seed"] = 10 + i
        generate_trace_file(params, trace_file_name)
        traces.append(trace_file_name)

    return traces


@timing
def get_raw_stats_for_policy(
    # basic simulation params
    base_params: dict,
    # files with traces to use as sim input
    trace_files: list,
    # policy to test, should be a policy key defined earlier
    policy_algorithm: str,
) -> List[SimulatorStats]:
    """Get full SimulatorStats for a given policy.

    Args:
        base_params: Basic simulation parameters
        trace_files: List of trace files to use as simulation input
        policy_algorithm: Policy key to test

    Returns:
        List of SimulatorStats objects (one per trace file, filtering out failed simulations)
    """
    params = base_params.copy()
    params["scheduler_algo"] = policy_algorithm
    # Run sequentially in this process to preserve scheduler registrations from exec()
    all_stats = [
        run_simulation_with_trace(params, trace_file) for trace_file in trace_files
    ]
    # Filter out None results (from simulations where no containers completed)
    stats = [s for s in all_stats if s is not None]
    return stats


def extract_metrics_from_stats(
    # full SimulatorStats from get_raw_stats_for_policy
    raw_stats: List[SimulatorStats],
    # metric to return: "latency" or "throughput"
    metric: str = "throughput",
) -> List[float]:
    """Extract metric values from SimulatorStats.

    Args:
        raw_stats: List of SimulatorStats from get_raw_stats_for_policy
        metric: Metric to return - either "latency" or "throughput"

    Returns:
        List of metric values (one per SimulatorStats)
    """
    if metric == "latency":
        return [s.adjusted_latency() for s in raw_stats]
    else:
        assert metric == "throughput", f"Unknown metric: {metric}"
        return [s.throughput for s in raw_stats]


def get_stats_for_policy(
    base_params: dict,
    trace_files: list,
    policy_algorithm: str,
    metric: str = "throughput",
) -> List[float]:
    """Run simulations and extract metric values for a given policy.

    This is a convenience function that combines get_raw_stats_for_policy
    and extract_metrics_from_stats.

    Args:
        base_params: Basic simulation parameters
        trace_files: List of trace files to use as simulation input
        policy_algorithm: Policy key to test
        metric: Metric to return - either "latency" or "throughput"

    Returns:
        List of metric values (one per trace file)
    """
    raw_stats = get_raw_stats_for_policy(base_params, trace_files, policy_algorithm)
    return extract_metrics_from_stats(raw_stats, metric)
