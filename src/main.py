# we need to import some modules because they are used in the policy-as-string
# but we mark them so that linters ignore the fact that they are unused here
from typing import List, Tuple  # noqa: F401
from dotenv import load_dotenv
import statistics

# eudoxia-specific imports
from eudoxia.simulator import get_param_defaults
from eudoxia.scheduler import register_scheduler_init, register_scheduler  # noqa: F401
from eudoxia.executor import Failure, Suspend, Assignment  # noqa: F401
from eudoxia.workload import WorkloadGenerator, Pipeline, Operator  # noqa: F401
from eudoxia.utils import Priority  # noqa: F401

# import utils
from simulation_utils import (
    generate_traces,
    get_stats_for_policy,
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
    generate_and_save_policy,
    setup_cost_tracking,
    reset_cost_tracking,
    get_cost_statistics,
)

# load env variables
load_dotenv()
# Ensure API keys to call the models are set
assert os.environ.get("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY not set in env"
assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY not set in env"


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


def main(
    n_traces: int = 3,
    duration: int = 60,
    ticks_per_second: int = 1000,
    num_pools: int = 1,
    cpu_pool: int = 64,
    ram_pool: int = 500,
    num_pipelines: int = 20,
    num_operators: int = 5,
    waiting_seconds_mean: float = 5,
    max_iterations: int = 5,
    verbose: bool = False,
    model: str = "claude-sonnet-4-5-20250929",
    temperature: float = 0.7,
    metric: str = "throughput",
):
    """Main function for policy generation and simulation.

    Args:
        n_traces: Number of trace files to generate
        duration: Simulation duration in seconds
        ticks_per_second: Number of simulation ticks per second
        num_pools: Number of resource pools
        cpu_pool: Number of CPUs per pool
        ram_pool: Amount of RAM per pool (GB)
        num_pipelines: Number of pipelines in the workload
        num_operators: Number of operators per pipeline
        waiting_seconds_mean: Mean waiting time in seconds for workload generation
        verbose: Whether to print detailed output
        max_iterations: Maximum number of policy generation iterations
        model: LLM model to use for policy generation
        temperature: Temperature parameter for LLM sampling (if 1.0 and model is gpt-5, uses high reasoning effort)
        metric: Metric to optimize - either "throughput" (higher is better) or "latency" (lower is better)
    """

    # Set up cost tracking
    setup_cost_tracking()
    reset_cost_tracking()

    # SIMULATION PARAMETERS
    base_params = get_param_defaults()
    base_params["duration"] = duration
    base_params["ticks_per_second"] = ticks_per_second
    base_params["num_pools"] = num_pools
    base_params["cpu_pool"] = cpu_pool
    base_params["ram_pool"] = ram_pool
    base_params["num_pipelines"] = num_pipelines
    base_params["num_operators"] = num_operators
    base_params["waiting_seconds_mean"] = waiting_seconds_mean
    base_params["interactive_prob"] = 0.3
    base_params["query_prob"] = 0.1
    base_params["batch_prob"] = 0.6

    print(f"Using parameters: {base_params}")

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
    naive_stats = get_stats_for_policy(base_params, trace_files, "naive", metric)
    baseline_median_metric = statistics.median(naive_stats)
    print_policy_stats("naive", naive_stats, prefix="Baseline", metric=metric)
    print("\n" + "=" * 60)

    # Initialize tracking for iterative improvement
    feedback_history = []
    best_policy_code = None
    best_policy_key = None
    best_median_metric = baseline_median_metric
    best_stats = naive_stats

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
        generated_policy_code, policy_filepath = generate_and_save_policy(
            user_request,
            verbose=verbose,
            feedback_history=feedback_history,
            model=model,
            temperature=temperature,
            policy_key=policy_key,
        )

        # Execute the generated policy code
        print(f"\nExecuting generated policy code (iteration {iteration + 1})...")
        try:
            exec(generated_policy_code)
            print("✅ Policy code is valid Python!")
            print(f"📝 Using scheduler key: {policy_key}")
        except Exception as e:
            print(f"❌ Policy execution failed: {e}")
            # Add error feedback
            feedback = f"The policy code resulted in a software error during execution: {e}\n\nPlease fix the error and generate a corrected policy."
            feedback_history.append(
                {"policy_code": generated_policy_code, "feedback": feedback}
            )
            # Record iteration time
            iteration_end_time = time.time()
            iteration_duration = iteration_end_time - iteration_start_time
            iteration_times.append(iteration_duration)
            print(f"⏱️  Iteration time: {iteration_duration:.5f}s")
            continue

        # Run the simulator with the new policy
        print(
            f"\nRunning simulator with policy: {policy_key} (saved to {policy_filepath})"
        )
        try:
            policy_stats = get_stats_for_policy(
                base_params, trace_files, policy_key, metric
            )
            median_metric = statistics.median(policy_stats)

            print_policy_stats(policy_key, policy_stats, metric=metric)

            # Compare with best policy (higher throughput is better, lower latency is better)
            if metric == "latency":
                is_better = median_metric < best_median_metric
            else:  # throughput
                is_better = median_metric > best_median_metric

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

        except Exception as e:
            print(f"❌ Simulation failed: {e}")
            feedback = f"The policy code executed but the simulation failed with error: {e}. Please generate a corrected policy."
            feedback_history.append(
                {"policy_code": generated_policy_code, "feedback": feedback}
            )
            # Record iteration time
            iteration_end_time = time.time()
            iteration_duration = iteration_end_time - iteration_start_time
            iteration_times.append(iteration_duration)
            print(f"⏱️  Iteration time: {iteration_duration:.5f}s")
            continue

    # Final summary
    print("\n" + "=" * 70)
    print("ITERATIVE OPTIMIZATION COMPLETE")
    print("=" * 70)

    # Print cost statistics
    cost_stats = get_cost_statistics()
    if cost_stats["num_requests"] > 0:
        print("\n💰 COST STATISTICS:")
        print(f"  Total Cost: ${cost_stats['total_cost']:.4f}")
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
        print(f"\nTo reuse this policy, load: policies/{best_policy_key}.py")
        print(f"Remember this run as: {best_policy_key}")
    else:
        print("\n⚠️  No successful policy was generated during iterations.")
        print("All attempts either failed to execute or did not improve over baseline.")


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
        "--cpu-pool", type=int, default=64, help="Number of CPUs per pool"
    )
    parser.add_argument(
        "--ram-pool", type=int, default=500, help="Amount of RAM per pool (GB)"
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
        default="throughput",
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
        ],
        default="gpt-5",
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
        cpu_pool=args.cpu_pool,
        ram_pool=args.ram_pool,
        num_pipelines=args.num_pipelines,
        num_operators=args.num_operators,
        waiting_seconds_mean=args.waiting_seconds_mean,
        max_iterations=args.max_iterations,
        verbose=args.verbose,
        model=args.model,
        temperature=args.temperature,
        metric=args.metric,
    )
