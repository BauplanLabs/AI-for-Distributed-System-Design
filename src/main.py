# we need to import some modules because they are used in the policy-as-string
# but we mark them so that linters ignore the fact that they are unused here
from typing import List, Tuple  # noqa: F401
from dotenv import load_dotenv
import statistics
import json
from pathlib import Path
from datetime import datetime

# eudoxia-specific imports
from eudoxia.workload import Pipeline, OperatorState  # noqa: F401
from eudoxia.workload.runtime_status import ASSIGNABLE_STATES  # noqa: F401
from eudoxia.executor.assignment import Assignment, ExecutionResult, Suspend  # noqa: F401
from eudoxia.scheduler.decorators import register_scheduler_init, register_scheduler  # noqa: F401
from eudoxia.utils import Priority  # noqa: F401
from eudoxia.simulator import get_param_defaults  # noqa: F401


# import utils
from simulation_utils import (
    generate_traces,
    get_raw_stats_for_policy,
    extract_metrics_from_stats,
)

# import prompts
from prompts import get_user_request

# logging setup - for some reason LLM libraries are very noisy
import os
import logging
import time
from wonderwords import RandomWord

logging.getLogger("eudoxia").setLevel(logging.CRITICAL)
logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
# import LLM functions after logging setup to avoid noisy output
from llm import (  # noqa: E402
    generate_policy,
    setup_cost_tracking,
    reset_cost_tracking,
    get_cost_statistics,
    get_last_request_cost,
)

# load env variables
load_dotenv()
# Ensure API keys to call the models are set
assert os.environ.get("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY not set in env"
assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY not set in env"


def convert_to_json_serializable(obj):
    """Recursively convert numpy types and other non-serializable types to JSON-serializable types."""
    import numpy as np
    import math

    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        val = float(obj)
        # Handle special float values that JSON doesn't support
        if math.isnan(val):
            return None
        elif math.isinf(val):
            return "Infinity" if val > 0 else "-Infinity"
        return val
    elif isinstance(obj, float):
        # Handle Python floats with special values
        if math.isnan(obj):
            return None
        elif math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        return obj
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return [convert_to_json_serializable(item) for item in obj.tolist()]
    elif hasattr(obj, "__dict__"):
        # Handle objects with __dict__ attribute
        return convert_to_json_serializable(obj.__dict__)
    else:
        return obj


def print_policy_stats(
    policy_name: str, stats: list, prefix: str = "", metric: str = "throughput"
):
    """Print statistics for a policy run.

    Args:
        policy_name: Name/key of the policy
        stats: List of metric values from get_stats_for_policy
        prefix: Optional prefix for the output (e.g., "Baseline")
        metric: The metric being measured (e.g., "throughput" or "latency")
    """
    metric_values = stats
    if prefix:
        print(f"\n=====> {prefix} '{policy_name}' policy stats:")
    else:
        print(f"\n=====> Policy stats for {policy_name}:")
    print(f"  {metric.capitalize()}: {metric_values}")
    print(f"  Median {metric}: {statistics.median(metric_values):.2f}")


def create_experiment_run_dir(model: str) -> Path:
    """Create experiment run directory with timestamp.

    Args:
        model: Model name to include in directory name

    Returns:
        Path to the created experiment run directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Sanitize model name for filesystem
    model_safe = model.replace("/", "_").replace(":", "_")
    run_dir = Path("src/experiment_runs") / f"{model_safe}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "iterations").mkdir(exist_ok=True)
    return run_dir


def save_run_config(
    run_dir: Path,
    base_params: dict,
    model: str,
    temperature: float,
    metric: str,
    max_iterations: int,
):
    """Save the run configuration to JSON.

    Args:
        run_dir: Path to experiment run directory
        base_params: Simulation parameters
        model: LLM model used
        temperature: Temperature parameter
        metric: Optimization metric
        max_iterations: Maximum number of iterations
    """
    config = {
        "model": model,
        "temperature": temperature,
        "metric": metric,
        "max_iterations": max_iterations,
        "simulation_params": base_params,
        "run_start_time": datetime.now().isoformat(),
    }
    with open(run_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)


def save_baseline_results(
    run_dir: Path, naive_stats: list, naive_raw_stats: list, metric: str
):
    """Save baseline results to JSON.

    Args:
        run_dir: Path to experiment run directory
        naive_stats: List of metric values for naive policy
        naive_raw_stats: Raw SimulatorStats objects
        metric: The metric being measured
    """
    baseline = {
        "policy": "naive",
        "metric": metric,
        "metric_values": naive_stats,
        "median_metric": statistics.median(naive_stats),
        "raw_stats": [s.to_dict() for s in naive_raw_stats],
    }
    # Convert entire baseline dict to JSON-serializable format
    baseline = convert_to_json_serializable(baseline)
    with open(run_dir / "baseline_results.json", "w") as f:
        json.dump(baseline, f, indent=2)


def save_iteration_data(
    run_dir: Path,
    iteration: int,
    policy_key: str,
    policy_result: dict,
    iteration_start_time: float,
    iteration_end_time: float,
    llm_cost: float,
    simulation_success: bool,
    policy_stats: list = None,
    policy_raw_stats: list = None,
    median_metric: float = None,
    baseline_median_metric: float = None,
    is_best: bool = False,
    metric: str = "latency",
    error_message: str = None,
):
    """Save all iteration data to a subfolder.

    Args:
        run_dir: Path to experiment run directory
        iteration: Iteration number (0-indexed)
        policy_key: Unique policy identifier
        policy_result: Dict from generate_policy with policy_code, llm_messages, llm_params
        iteration_start_time: Timestamp when iteration started
        iteration_end_time: Timestamp when iteration ended
        llm_cost: Cost of LLM call for this iteration
        simulation_success: Whether simulation ran successfully
        policy_stats: List of metric values (if simulation succeeded)
        policy_raw_stats: Raw SimulatorStats objects (if simulation succeeded)
        median_metric: Median metric value (if simulation succeeded)
        baseline_median_metric: Baseline median metric for comparison
        is_best: Whether this is the best policy so far
        metric: The metric being measured
        error_message: Error message if iteration failed
    """
    iter_dir = run_dir / "iterations" / policy_key
    iter_dir.mkdir(parents=True, exist_ok=True)

    # Save policy code
    with open(iter_dir / "policy.py", "w") as f:
        f.write(policy_result["policy_code"])

    # Save LLM context (messages sent to LLM)
    with open(iter_dir / "llm_context.json", "w") as f:
        json.dump(
            {
                "messages": policy_result["llm_messages"],
                "params": policy_result["llm_params"],
            },
            f,
            indent=2,
        )

    # Save iteration metadata
    iteration_duration = iteration_end_time - iteration_start_time
    metadata = {
        "iteration": iteration + 1,
        "policy_key": policy_key,
        "model": policy_result["llm_params"]["model"],
        "temperature": policy_result["llm_params"]["temperature"],
        "reasoning_effort": policy_result["llm_params"]["reasoning_effort"],
        "llm_cost": llm_cost,
        "start_time": datetime.fromtimestamp(iteration_start_time).isoformat(),
        "end_time": datetime.fromtimestamp(iteration_end_time).isoformat(),
        "duration_seconds": iteration_duration,
        "simulation_success": simulation_success,
        "error_message": error_message,
    }
    with open(iter_dir / "iteration_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save simulation results (if available)
    if simulation_success and policy_stats is not None:
        improvement_pct = None
        if baseline_median_metric and baseline_median_metric != 0:
            improvement_pct = (
                (median_metric - baseline_median_metric) / baseline_median_metric
            ) * 100

        results = {
            "metric": metric,
            "metric_values": policy_stats,
            "median_metric": median_metric,
            "baseline_median_metric": baseline_median_metric,
            "improvement_pct": improvement_pct,
            "is_best_so_far": is_best,
            "raw_stats": [s.to_dict() for s in policy_raw_stats]
            if policy_raw_stats
            else None,
        }
        # Convert entire results dict to JSON-serializable format
        results = convert_to_json_serializable(results)
        with open(iter_dir / "simulation_results.json", "w") as f:
            json.dump(results, f, indent=2)


def save_run_summary(
    run_dir: Path,
    iteration_results: list,
    best_policy_key: str,
    best_median_metric: float,
    baseline_median_metric: float,
    total_time: float,
    total_cost: float,
    metric: str,
    model: str,
):
    """Save run summary for table/chart generation.

    Args:
        run_dir: Path to experiment run directory
        iteration_results: List of dicts with per-iteration summary data
        best_policy_key: Key of the best performing policy
        best_median_metric: Best metric value achieved
        baseline_median_metric: Baseline metric value
        total_time: Total run time in seconds
        total_cost: Total LLM cost
        metric: The metric being measured
        model: LLM model used
    """
    improvement_pct = None
    if baseline_median_metric and baseline_median_metric != 0:
        improvement_pct = (
            (best_median_metric - baseline_median_metric) / baseline_median_metric
        ) * 100

    summary = {
        "model": model,
        "metric": metric,
        "baseline_median_metric": baseline_median_metric,
        "best_policy_key": best_policy_key,
        "best_median_metric": best_median_metric,
        "improvement_pct": improvement_pct,
        "total_iterations": len(iteration_results),
        "successful_iterations": sum(
            1 for r in iteration_results if r.get("simulation_success", False)
        ),
        "total_time_seconds": total_time,
        "total_cost": total_cost,
        "run_end_time": datetime.now().isoformat(),
        "iteration_timeseries": iteration_results,
    }
    # Convert entire summary dict to JSON-serializable format
    summary = convert_to_json_serializable(summary)
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def main(
    n_traces: int = 3,
    duration: int = 60,
    ticks_per_second: int = 1000,
    num_pools: int = 1,
    cpus_per_pool: int = 64,
    ram_gb_per_pool: int = 500,
    num_pipelines: int = 20,
    num_operators: int = 5,
    waiting_seconds_mean: float = 5,
    multi_operator_containers: bool = False,
    max_iterations: int = 5,
    verbose: bool = False,
    model: str = "claude-sonnet-4-5-20250929",
    temperature: float = 0.7,
    metric: str = "latency",
):
    """Main function for policy generation and simulation.

    Args:
        n_traces: Number of trace files to generate
        duration: Simulation duration in seconds
        ticks_per_second: Number of simulation ticks per second
        num_pools: Number of resource pools
        cpus_per_pool: Number of CPUs per pool
        ram_gb_per_pool: Amount of RAM per pool (GB)
        num_pipelines: Number of pipelines in the workload
        num_operators: Number of operators per pipeline
        waiting_seconds_mean: Mean waiting time in seconds for workload generation
        multi_operator_containers: Whether to allow multiple operators per container
        verbose: Whether to print detailed output
        max_iterations: Maximum number of policy generation iterations
        model: LLM model to use for policy generation
        temperature: Temperature parameter for LLM sampling (if 1.0 and model is gpt-5, uses high reasoning effort)
        metric: Metric to optimize - either "throughput" (higher is better) or "latency" (lower is better)
    """

    # Set up cost tracking
    setup_cost_tracking()
    reset_cost_tracking()

    # Create experiment run directory
    run_dir = create_experiment_run_dir(model)
    print(f"\n📁 Experiment run directory: {run_dir}")

    # SIMULATION PARAMETERS
    base_params = get_param_defaults()
    base_params["duration"] = duration
    base_params["ticks_per_second"] = ticks_per_second
    base_params["num_pools"] = num_pools
    base_params["cpus_per_pool"] = cpus_per_pool
    base_params["ram_gb_per_pool"] = ram_gb_per_pool
    base_params["num_pipelines"] = num_pipelines
    base_params["num_operators"] = num_operators
    base_params["waiting_seconds_mean"] = waiting_seconds_mean
    base_params["multi_operator_containers"] = multi_operator_containers
    base_params["interactive_prob"] = 0.3
    base_params["query_prob"] = 0.1
    base_params["batch_prob"] = 0.6

    print(f"Using parameters: {base_params}")

    # Save run configuration
    save_run_config(run_dir, base_params, model, temperature, metric, max_iterations)

    # SETUP TRACES ONCE (IT'S EXPENSIVE)
    # Generate traces in 3 batches with varying parameters for diversity
    trace_files = []
    original_num_pipelines = num_pipelines
    original_waiting_seconds_mean = waiting_seconds_mean

    for batch_idx in range(n_traces):
        # Scale parameters by 2^batch_idx (1x, 2x, 4x)
        scale_factor = 2**batch_idx
        batch_params = base_params.copy()
        batch_params["num_pipelines"] = original_num_pipelines * scale_factor
        batch_params["waiting_seconds_mean"] = (
            original_waiting_seconds_mean * scale_factor
        )
        # Generate 2 traces for this batch
        file_name_prefix = f"trace_scale{scale_factor}x_{base_params['duration']}s"
        batch_trace_files = generate_traces(
            k=2,
            base_params=batch_params,
            file_name_prefix=file_name_prefix,
        )
        trace_files.extend(batch_trace_files)
    assert len(trace_files) == len(set(trace_files))

    print(
        f"\nGenerated {len(trace_files)} total trace files across {n_traces} parameter configurations"
    )
    print("Traces:", ", ".join(trace_files))

    # Establish a baseline result by running policies
    print("\nRunning baseline policies...")
    print("First, running 'naive' policy...")
    naive_raw_stats = get_raw_stats_for_policy(base_params, trace_files, "naive")
    naive_stats = extract_metrics_from_stats(naive_raw_stats, metric)
    baseline_median_metric = statistics.median(naive_stats)
    print_policy_stats("naive", naive_stats, prefix="Baseline", metric=metric)

    # Save baseline results
    save_baseline_results(run_dir, naive_stats, naive_raw_stats, metric)
    print("\n" + "=" * 60)

    # Initialize tracking for iterative improvement
    feedback_history = []
    best_policy_code = None
    best_policy_key = None
    best_median_metric = baseline_median_metric
    best_stats = naive_stats

    # Track iteration results for timeseries/summary
    iteration_results = []

    print("\n" + "=" * 60)
    print(f"STARTING ITERATIVE POLICY GENERATION (max {max_iterations} iterations)")
    print("=" * 60)

    # Track iteration timing
    iteration_times = []
    total_start_time = time.time()

    # Iterative policy generation loop
    for iteration in range(max_iterations):
        iteration_start_time = time.time()
        print(f"\n{'=' * 60}")
        print(f"ITERATION {iteration + 1}/{max_iterations}")
        print("=" * 60)

        # Generate a unique policy key for this iteration
        r = RandomWord()
        adjective = r.word(include_parts_of_speech=["adjectives"])
        noun = r.word(include_parts_of_speech=["nouns"])
        policy_key = f"{adjective}_{noun}_iter{iteration + 1}"
        print(f"Generated policy key: {policy_key}")

        # Generate policy with feedback from previous attempts
        user_request = get_user_request(policy_key, metric)
        policy_result = generate_policy(
            user_request,
            verbose=verbose,
            feedback_history=feedback_history,
            model=model,
            temperature=temperature,
            policy_key=policy_key,
        )
        generated_policy_code = policy_result["policy_code"]

        # Get the cost of this LLM call
        llm_cost = get_last_request_cost()

        # Execute the generated policy code
        print(f"\nExecuting generated policy code (iteration {iteration + 1})...")
        try:
            exec(generated_policy_code, globals())
            print("✅ Policy code is valid Python!")
            print(f"📝 Using scheduler key: {policy_key}")
        except Exception as e:
            print(f"❌ Policy execution failed: {e}")
            # Add error feedback
            error_msg = str(e)
            feedback = f"The policy code resulted in a software error during execution: {e}\n\nPlease fix the error and generate a corrected policy."
            feedback_history.append(
                {"policy_code": generated_policy_code, "feedback": feedback}
            )
            # Record iteration time
            iteration_end_time = time.time()
            iteration_duration = iteration_end_time - iteration_start_time
            iteration_times.append(iteration_duration)
            print(f"⏱️  Iteration time: {iteration_duration:.5f}s")

            # Save iteration data for failed execution
            save_iteration_data(
                run_dir=run_dir,
                iteration=iteration,
                policy_key=policy_key,
                policy_result=policy_result,
                iteration_start_time=iteration_start_time,
                iteration_end_time=iteration_end_time,
                llm_cost=llm_cost,
                simulation_success=False,
                baseline_median_metric=baseline_median_metric,
                metric=metric,
                error_message=f"Execution error: {error_msg}",
            )
            # Track for timeseries
            iteration_results.append(
                {
                    "iteration": iteration + 1,
                    "policy_key": policy_key,
                    "simulation_success": False,
                    "median_metric": None,
                    "is_best": False,
                    "llm_cost": llm_cost,
                    "duration_seconds": iteration_duration,
                    "error": f"Execution error: {error_msg}",
                }
            )
            continue

        # Run the simulator with the new policy
        print(f"\nRunning simulator with policy: {policy_key}")
        try:
            policy_raw_stats = get_raw_stats_for_policy(
                base_params, trace_files, policy_key
            )
            policy_stats = extract_metrics_from_stats(policy_raw_stats, metric)
            median_metric = statistics.median(policy_stats)

            print_policy_stats(policy_key, policy_stats, metric=metric)

            # Compare with best policy (higher throughput is better, lower latency is better)
            if metric == "latency":
                is_better = median_metric < best_median_metric
            else:  # throughput
                is_better = median_metric > best_median_metric

            # TODO: Include JSON stats in feedback via [s.to_dict() for s in policy_raw_stats]
            if is_better:
                improvement_direction = (
                    "decreased" if metric == "latency" else "improved"
                )
                print(
                    f"\n🎉 NEW BEST POLICY! {metric.capitalize()} {improvement_direction} from {best_median_metric:.2f} to {median_metric:.2f}"
                )
                best_policy_code = generated_policy_code
                best_policy_key = policy_key
                best_median_metric = median_metric
                best_stats = policy_stats

                feedback = f"Great! This policy achieved {metric} of {median_metric:.2f} (vs previous best: {best_median_metric:.2f}). This is an improvement! Can you further optimize it?"
            else:
                improvement_direction = "increase" if metric == "latency" else "improve"
                print(
                    f"\n📊 Policy {metric} ({median_metric:.2f}) did not {improvement_direction} over current best ({best_median_metric:.2f})"
                )
                feedback = f"The policy achieved {metric} of {median_metric:.2f}, which did not improve over the current best ({metric}: {best_median_metric:.2f}). Please try a different approach to improve performance."

            feedback_history.append(
                {"policy_code": generated_policy_code, "feedback": feedback}
            )

            # Record iteration time
            iteration_end_time = time.time()
            iteration_duration = iteration_end_time - iteration_start_time
            iteration_times.append(iteration_duration)
            print(f"⏱️  Iteration time: {iteration_duration:.5f}s")

            # Save iteration data for successful simulation
            save_iteration_data(
                run_dir=run_dir,
                iteration=iteration,
                policy_key=policy_key,
                policy_result=policy_result,
                iteration_start_time=iteration_start_time,
                iteration_end_time=iteration_end_time,
                llm_cost=llm_cost,
                simulation_success=True,
                policy_stats=policy_stats,
                policy_raw_stats=policy_raw_stats,
                median_metric=median_metric,
                baseline_median_metric=baseline_median_metric,
                is_best=is_better,
                metric=metric,
            )
            # Track for timeseries
            iteration_results.append(
                {
                    "iteration": iteration + 1,
                    "policy_key": policy_key,
                    "simulation_success": True,
                    "median_metric": median_metric,
                    "is_best": is_better,
                    "llm_cost": llm_cost,
                    "duration_seconds": iteration_duration,
                }
            )

        except Exception as e:
            print(f"❌ Simulation failed: {e}")
            error_msg = str(e)
            feedback = f"The policy code executed but the simulation failed with error: {e}. Please generate a corrected policy."
            feedback_history.append(
                {"policy_code": generated_policy_code, "feedback": feedback}
            )
            # Record iteration time
            iteration_end_time = time.time()
            iteration_duration = iteration_end_time - iteration_start_time
            iteration_times.append(iteration_duration)
            print(f"⏱️  Iteration time: {iteration_duration:.5f}s")

            # Save iteration data for failed simulation
            save_iteration_data(
                run_dir=run_dir,
                iteration=iteration,
                policy_key=policy_key,
                policy_result=policy_result,
                iteration_start_time=iteration_start_time,
                iteration_end_time=iteration_end_time,
                llm_cost=llm_cost,
                simulation_success=False,
                baseline_median_metric=baseline_median_metric,
                metric=metric,
                error_message=f"Simulation error: {error_msg}",
            )
            # Track for timeseries
            iteration_results.append(
                {
                    "iteration": iteration + 1,
                    "policy_key": policy_key,
                    "simulation_success": False,
                    "median_metric": None,
                    "is_best": False,
                    "llm_cost": llm_cost,
                    "duration_seconds": iteration_duration,
                    "error": f"Simulation error: {error_msg}",
                }
            )
            continue

    # Final summary
    print("\n" + "=" * 70)
    print("ITERATIVE OPTIMIZATION COMPLETE")
    print("=" * 70)

    # Print cost statistics
    cost_stats = get_cost_statistics()
    total_cost = cost_stats["total_cost"]
    if cost_stats["num_requests"] > 0:
        print("\n💰 COST STATISTICS:")
        print(f"  Total Cost: ${total_cost:.4f}")
        print(f"  Median Cost per Request: ${cost_stats['median_cost']:.4f}")
        print(f"  Number of Requests: {cost_stats['num_requests']}")

    # Print timing statistics
    total_time = time.time() - total_start_time
    if iteration_times:
        median_iteration_time = statistics.median(iteration_times)
        print("\n⏱️  TIMING STATISTICS:")
        print(
            f"  Total Iteration Time: {total_time:.5f}s ({total_time / 60:.5f} minutes)"
        )
        print(f"  Median Time per Iteration: {median_iteration_time:.5f}s")
        print(f"  Number of Iterations Completed: {len(iteration_times)}")

    if best_policy_code:
        print(f"\n🔑 BEST POLICY KEY: {best_policy_key}")
        print("\n🏆 BEST PERFORMANCE:")
        print_policy_stats(best_policy_key, best_stats, metric=metric)
        print("\n📈 IMPROVEMENT OVER BASELINE:")
        change_pct = (
            (best_median_metric - baseline_median_metric) / baseline_median_metric * 100
        )
        direction = "decrease" if metric == "latency" else "increase"
        print(
            f"  {metric.capitalize()}: {baseline_median_metric:.2f} → {best_median_metric:.2f} ({change_pct:.1f}% {direction})"
        )
        print("=" * 70)
        print(
            f"\nTo reuse this policy, see: {run_dir}/iterations/{best_policy_key}/policy.py"
        )
        print(f"Remember this run as: {best_policy_key}")
    else:
        print("\n⚠️  No successful policy was generated during iterations.")
        print("All attempts either failed to execute or did not improve over baseline.")

    # Save run summary for table/chart generation
    save_run_summary(
        run_dir=run_dir,
        iteration_results=iteration_results,
        best_policy_key=best_policy_key,
        best_median_metric=best_median_metric,
        baseline_median_metric=baseline_median_metric,
        total_time=total_time,
        total_cost=total_cost,
        metric=metric,
        model=model,
    )
    print(f"\n📊 Experiment data saved to: {run_dir}")
    print("   - run_config.json: Run configuration")
    print("   - baseline_results.json: Baseline (naive) policy results")
    print(
        "   - iterations/: Per-iteration data (policy, LLM context, metadata, results)"
    )
    print("   - summary.json: Summary for table/chart generation")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate and test scheduling policies using LLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Simulation parameters
    parser.add_argument(
        "--n-traces", type=int, default=5, help="Number of trace files to generate"
    )
    parser.add_argument(
        "--duration", type=int, default=600, help="Simulation duration in seconds"
    )
    parser.add_argument(
        "--ticks-per-second",
        type=int,
        default=1000,
        help="Number of simulation ticks per second",
    )
    parser.add_argument(
        "--num-pools", type=int, default=1, help="Number of resource pools"
    )
    parser.add_argument(
        "--cpus-per-pool", type=int, default=64, help="Number of CPUs per pool"
    )
    parser.add_argument(
        "--ram-gb-per-pool", type=int, default=500, help="Amount of RAM per pool (GB)"
    )
    parser.add_argument(
        "--num-pipelines",
        type=int,
        default=10,
        help="Number of pipelines in the workload",
    )
    parser.add_argument(
        "--num-operators", type=int, default=10, help="Number of operators per pipeline"
    )
    parser.add_argument(
        "--waiting-seconds-mean",
        type=float,
        default=10.0,
        help="Mean waiting time in seconds for workload generation",
    )
    parser.add_argument(
        "--multi-operator-containers",
        action="store_true",
        help="Allow multiple operators per container (default: false)",
    )
    # experimentation parameters
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=50,
        help="Maximum number of policy generation iterations",
    )
    parser.add_argument(
        "--metric",
        choices=["throughput", "latency"],
        default="latency",
        help="Metric to optimize - 'throughput' (higher is better) or 'latency' (lower is better)",
    )
    # LLM parameters
    parser.add_argument(
        "--model",
        choices=[
            "claude-opus-4-5-20251101",
            "claude-sonnet-4-5-20250929",
            "gpt-5",
            "gpt-5-mini",
            "gpt-5.2-2025-12-11",
        ],
        default="gpt-5.2-2025-12-11",
        help="LLM model to use for policy generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature parameter for LLM sampling (if model is from gpt-5 family, it gets ignored)",
    )
    # add verbose flag to enable detailed output
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    # Call main with explicit typed arguments
    main(
        n_traces=args.n_traces,
        duration=args.duration,
        ticks_per_second=args.ticks_per_second,
        num_pools=args.num_pools,
        cpus_per_pool=args.cpus_per_pool,
        ram_gb_per_pool=args.ram_gb_per_pool,
        num_pipelines=args.num_pipelines,
        num_operators=args.num_operators,
        waiting_seconds_mean=args.waiting_seconds_mean,
        multi_operator_containers=args.multi_operator_containers,
        max_iterations=args.max_iterations,
        verbose=args.verbose,
        model=args.model,
        temperature=args.temperature,
        metric=args.metric,
    )
