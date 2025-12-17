"""
RL Training for Scheduling Policy Generation using Tinker.

Uses reinforcement learning to train an LLM to generate scheduling policies
for the Eudoxia simulator. The simulation provides continuous rewards based
on throughput or latency metrics.

Usage:
    uv run python src/training.py
    uv run python src/training.py metric=latency
    uv run python src/training.py model_name='openai/gpt-oss-20b' batch_size=4
"""

import json
import logging
import os
import re
import statistics
import time
from pathlib import Path

import chz
import tinker
import torch
from dotenv import load_dotenv
from tinker import types
from tinker.types.tensor_data import TensorData

# eudoxia imports for simulation
from eudoxia.simulator import get_param_defaults
from eudoxia.scheduler import register_scheduler_init, register_scheduler  # noqa: F401
from eudoxia.executor import Failure, Suspend, Assignment  # noqa: F401
from eudoxia.workload import WorkloadGenerator, Pipeline, Operator  # noqa: F401
from eudoxia.utils import Priority  # noqa: F401

# local imports
from simulation_utils import generate_traces, get_stats_for_policy
from prompts import POLICY_GENERATION_SYSTEM_PROMPT
from llm import build_system_context, clean_generated_code

# Load environment variables
load_dotenv(Path(__file__).parent / ".env")

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)
logging.getLogger("eudoxia").setLevel(logging.CRITICAL)

# Default log path
DEFAULT_LOG_PATH = str(Path(__file__).parent / "training_logs")


def _default_trace_configs() -> list[dict]:
    """
    Default trace configurations with diverse parameter variations.

    Varies multiple dimensions to expose the model to different scenarios:
    - Workload intensity: num_pipelines, waiting_seconds_mean
    - Workload complexity: num_operators per pipeline
    - Priority distribution: interactive_prob, query_prob, batch_prob
    - Resource constraints: cpu_pool, ram_pool

    Returns a list of ~23 configurations covering different workload profiles.
    With default traces_per_config=2, this generates 46 trace files.
    """
    configs = []

    # Baseline configuration
    baseline = {
        "num_pipelines": 15,
        "waiting_seconds_mean": 10.0,
        "num_operators": 8,
        "interactive_prob": 0.3,
        "query_prob": 0.1,
        "batch_prob": 0.6,
        "cpu_pool": 64,
        "ram_pool": 500,
    }

    # 1. Vary workload intensity (4 configs)
    intensity_levels = [
        {"num_pipelines": 5, "waiting_seconds_mean": 5.0, "label": "light"},
        {"num_pipelines": 15, "waiting_seconds_mean": 10.0, "label": "medium"},
        {"num_pipelines": 30, "waiting_seconds_mean": 15.0, "label": "heavy"},
        {"num_pipelines": 50, "waiting_seconds_mean": 20.0, "label": "intense"},
    ]
    for intensity in intensity_levels:
        config = baseline.copy()
        config["num_pipelines"] = intensity["num_pipelines"]
        config["waiting_seconds_mean"] = intensity["waiting_seconds_mean"]
        config["config_label"] = f"intensity_{intensity['label']}"
        configs.append(config)

    # 2. Vary operator complexity (3 configs)
    for num_ops, label in [(3, "simple"), (8, "moderate"), (15, "complex")]:
        config = baseline.copy()
        config["num_operators"] = num_ops
        config["config_label"] = f"operators_{label}"
        configs.append(config)

    # 3. Vary priority distribution (4 configs)
    priority_profiles = [
        {
            "interactive_prob": 0.1,
            "query_prob": 0.1,
            "batch_prob": 0.8,
            "label": "batch_heavy",
        },
        {
            "interactive_prob": 0.3,
            "query_prob": 0.2,
            "batch_prob": 0.5,
            "label": "balanced",
        },
        {
            "interactive_prob": 0.5,
            "query_prob": 0.3,
            "batch_prob": 0.2,
            "label": "interactive_heavy",
        },
        {
            "interactive_prob": 0.2,
            "query_prob": 0.6,
            "batch_prob": 0.2,
            "label": "query_heavy",
        },
    ]
    for prio in priority_profiles:
        config = baseline.copy()
        config["interactive_prob"] = prio["interactive_prob"]
        config["query_prob"] = prio["query_prob"]
        config["batch_prob"] = prio["batch_prob"]
        config["config_label"] = f"priority_{prio['label']}"
        configs.append(config)

    # 4. Vary resource constraints (3 configs)
    for cpu, ram, label in [
        (32, 256, "constrained"),
        (64, 500, "standard"),
        (128, 1000, "generous"),
    ]:
        config = baseline.copy()
        config["cpu_pool"] = cpu
        config["ram_pool"] = ram
        config["config_label"] = f"resources_{label}"
        configs.append(config)

    # 5. Cross-combinations: high intensity + different priorities (4 configs)
    for prio in priority_profiles:
        config = baseline.copy()
        config["num_pipelines"] = 40
        config["waiting_seconds_mean"] = 18.0
        config["interactive_prob"] = prio["interactive_prob"]
        config["query_prob"] = prio["query_prob"]
        config["batch_prob"] = prio["batch_prob"]
        config["config_label"] = f"high_intensity_{prio['label']}"
        configs.append(config)

    # 6. Cross-combinations: complex operators + resource variations (2 configs)
    for cpu, ram, label in [(32, 256, "constrained"), (64, 500, "standard")]:
        config = baseline.copy()
        config["num_operators"] = 12
        config["cpu_pool"] = cpu
        config["ram_pool"] = ram
        config["config_label"] = f"complex_ops_{label}_resources"
        configs.append(config)

    # 7. Stress test scenarios (3 configs)
    configs.append(
        {
            **baseline,
            "num_pipelines": 60,
            "waiting_seconds_mean": 5.0,  # Many pipelines, fast arrival
            "config_label": "stress_high_concurrency",
        }
    )
    configs.append(
        {
            **baseline,
            "num_pipelines": 20,
            "num_operators": 20,
            "cpu_pool": 32,
            "ram_pool": 256,
            "config_label": "stress_complex_constrained",
        }
    )
    configs.append(
        {
            **baseline,
            "num_pipelines": 40,
            "interactive_prob": 0.6,
            "query_prob": 0.3,
            "batch_prob": 0.1,
            "config_label": "stress_latency_sensitive",
        }
    )

    return configs  # Total: 23 configurations


@chz.chz
class Config:
    """Training configuration."""

    base_url: str | None = None
    log_path: str = DEFAULT_LOG_PATH
    model_name: str = "openai/gpt-oss-20b"
    batch_size: int = 4
    group_size: int = 4
    learning_rate: float = 1e-5
    lora_rank: int = 32
    save_every: int = 10
    max_tokens: int = 4096
    num_epochs: int = 1
    # Simulation parameters
    metric: str = "throughput"  # "throughput" or "latency"
    duration: int = 60
    ticks_per_second: int = 1000
    num_pools: int = 1
    cpu_pool: int = 64
    ram_pool: int = 500
    num_operators: int = 8  # Default operators per pipeline
    traces_per_config: int = 2  # With 23 configs, generates 46 traces by default
    trace_configs: list[dict] = chz.field(default_factory=_default_trace_configs)


def get_base_params(config: Config) -> dict:
    """Build base simulation parameters from config."""
    base_params = get_param_defaults()
    base_params["duration"] = config.duration
    base_params["ticks_per_second"] = config.ticks_per_second
    base_params["num_pools"] = config.num_pools
    base_params["cpu_pool"] = config.cpu_pool
    base_params["ram_pool"] = config.ram_pool
    base_params["num_operators"] = config.num_operators
    base_params["interactive_prob"] = 0.3
    base_params["query_prob"] = 0.1
    base_params["batch_prob"] = 0.6
    return base_params


def generate_training_traces(config: Config) -> list[str]:
    """
    Generate a diverse set of trace files for training.

    Creates traces with varying parameters across multiple dimensions to expose
    the model to different workload scenarios. Each trace configuration can vary:
    - num_pipelines, waiting_seconds_mean (workload intensity)
    - num_operators (complexity)
    - interactive_prob, query_prob, batch_prob (priority mix)
    - cpu_pool, ram_pool (resource constraints)
    """
    base_params = get_base_params(config)
    trace_files = []

    for idx, trace_config in enumerate(config.trace_configs):
        batch_params = base_params.copy()

        # Apply all parameters from the trace config
        param_keys = [
            "num_pipelines",
            "waiting_seconds_mean",
            "num_operators",
            "interactive_prob",
            "query_prob",
            "batch_prob",
            "cpu_pool",
            "ram_pool",
        ]
        for key in param_keys:
            if key in trace_config:
                batch_params[key] = trace_config[key]

        # Create descriptive file name prefix
        config_label = trace_config.get("config_label", f"config_{idx}")
        file_name_prefix = f"train_trace_{config_label}_{config.duration}s"

        batch_trace_files = generate_traces(
            k=config.traces_per_config,
            base_params=batch_params,
            file_name_prefix=file_name_prefix,
        )
        trace_files.extend(batch_trace_files)

    assert len(trace_files) == len(set(trace_files)), "Duplicate trace files generated"
    logger.info(
        f"Generated {len(trace_files)} traces from {len(config.trace_configs)} configurations"
    )
    return trace_files


def build_prompt(policy_key: str, metric: str, system_context: str) -> str:
    """Build the full prompt for policy generation."""
    system_prompt = POLICY_GENERATION_SYSTEM_PROMPT.format(context=system_context)

    user_request = f"""
Starting from the naive policy provided as example, generate a novel scheduling policy
that optimizes for {metric}. Focus on improving {metric} by leveraging concepts like
priority-based scheduling, resource allocation strategies, and intelligent queue management.

IMPORTANT: Use the following EXACT key in both @register_scheduler_init and @register_scheduler decorators: "{policy_key}"
Do NOT generate your own key - you MUST use exactly: "{policy_key}"

COMMON MISTAKES TO AVOID:
1. Operator objects do NOT have .cpu or .ram attributes. Do not try to access op.cpu or op.ram.
2. CPU and RAM are allocated at the Assignment level, not per-operator. Get operators via: ops = [op for op in pipeline.values]
3. Assignment constructor REQUIRES pipeline_id: Assignment(ops=op_list, cpu=cpu_amount, ram=ram_amount, priority=priority, pool_id=pool_id, pipeline_id=p.pipeline_id)
4. Pool resources: s.executor.pools[pool_id].avail_cpu_pool, .avail_ram_pool, .max_cpu_pool, .max_ram_pool
5. Pipeline attributes: p.values (list of operators), p.priority (Priority enum), p.pipeline_id (string ID)
6. Priority levels: Priority.QUERY (highest), Priority.INTERACTIVE, Priority.BATCH_PIPELINE (lowest)
7. Failure object: f.ops, f.priority, f.pool_id, f.cpu, f.ram, f.container_id, f.error
8. Suspend constructor: Suspend(container_id, pool_id)
9. CRITICAL RESOURCE TRACKING: avail_cpu_pool and avail_ram_pool do NOT update during your scheduling call. If you assign multiple pipelines, you MUST track how much you've already assigned. Example: track "assigned_cpu[pool_id]" and check "avail_cpu - assigned_cpu[pool_id]" for remaining resources.
10. Do NOT use type hints like List[...] or Tuple[...] - these are not imported. Just use plain Python.

First, briefly reason about your approach inside <reasoning> tags.
Then, output the complete Python code inside <code> tags.

Example format:
<reasoning>
I will improve throughput by prioritizing high-priority jobs and using efficient resource allocation...
</reasoning>

<code>
@register_scheduler_init(key="{policy_key}")
def init_scheduler(s):
    s.waiting_queue = []

@register_scheduler(key="{policy_key}")
def scheduler(s, failures, pipelines):
    suspensions = []
    assignments = []
    # Track resources we've assigned this call (avail_* doesn't update during the call)
    assigned_cpu = {{pool_id: 0 for pool_id in range(s.executor.num_pools)}}
    assigned_ram = {{pool_id: 0 for pool_id in range(s.executor.num_pools)}}

    for p in pipelines:
        ops = [op for op in p.values]
        for pool_id in range(s.executor.num_pools):
            avail_cpu = s.executor.pools[pool_id].avail_cpu_pool - assigned_cpu[pool_id]
            avail_ram = s.executor.pools[pool_id].avail_ram_pool - assigned_ram[pool_id]
            # Allocate a reasonable amount (e.g., 8 CPU, 64 RAM per pipeline)
            cpu_to_assign = min(8, avail_cpu)
            ram_to_assign = min(64, avail_ram)
            if cpu_to_assign > 0 and ram_to_assign > 0:
                assignment = Assignment(ops=ops, cpu=cpu_to_assign, ram=ram_to_assign,
                                        priority=p.priority, pool_id=pool_id,
                                        pipeline_id=p.pipeline_id)
                assignments.append(assignment)
                assigned_cpu[pool_id] += cpu_to_assign
                assigned_ram[pool_id] += ram_to_assign
                break
    return suspensions, assignments
</code>
""".strip()

    return f"{system_prompt}\n\n{user_request}"


def parse_policy_code(response_text: str) -> str | None:
    """
    Extract Python policy code from model response.

    Looks for code in <code> tags first, then falls back to markdown code blocks.
    Takes the LAST match to handle cases where model outputs multiple attempts.
    """
    response_text = response_text.strip()

    # Try to extract from <code> tags first (preferred format)
    code_matches = list(re.finditer(r"<code>(.*?)</code>", response_text, re.DOTALL))
    if code_matches:
        # Take the last match (final attempt)
        return code_matches[-1].group(1).strip()

    # Fall back to markdown code blocks
    code_match = re.search(r"```(?:python)?\n(.*?)```", response_text, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()

    # If no code blocks, assume the entire response is code
    # Check if it looks like valid Python policy code
    if "@register_scheduler" in response_text:
        return response_text.strip()

    return None


def execute_policy_and_get_reward(
    policy_code: str,
    policy_key: str,
    base_params: dict,
    trace_files: list[str],
    metric: str,
    baseline_metric: float,
) -> tuple[float, list[float] | None, str | None]:
    """
    Execute the generated policy and compute reward from simulation.

    Args:
        policy_code: Generated Python code for the policy
        policy_key: Unique key for this policy
        base_params: Base simulation parameters
        trace_files: List of trace files to run simulations on
        metric: "throughput" or "latency"
        baseline_metric: Baseline metric value for normalization

    Returns:
        Tuple of (reward, metric_values, error_message)
        - reward: Normalized reward value (higher is better, typically 0-2 range)
        - metric_values: Raw metric values from simulation (None if execution failed)
        - error_message: Error message if execution failed (None if successful)
    """
    # First, try to execute the policy code to register it
    try:
        exec(policy_code, globals())
    except Exception as e:
        return 0.0, None, f"Policy execution error: {e}"

    # Run simulation with the policy
    try:
        metric_values = get_stats_for_policy(
            base_params, trace_files, policy_key, metric
        )
        median_metric = statistics.median(metric_values)

        # Compute normalized reward
        # For throughput: higher is better, reward = metric / baseline
        # For latency: lower is better, reward = baseline / metric
        if metric == "latency":
            # Avoid division by zero
            if median_metric <= 0:
                reward = 0.0
            else:
                reward = baseline_metric / median_metric
        else:  # throughput
            if baseline_metric <= 0:
                reward = 1.0 if median_metric > 0 else 0.0
            else:
                reward = median_metric / baseline_metric

        return reward, metric_values, None

    except Exception as e:
        return 0.0, None, f"Simulation error: {e}"


def compute_baseline_metric(
    base_params: dict,
    trace_files: list[str],
    metric: str,
) -> float:
    """Compute baseline metric using the naive scheduler."""
    naive_stats = get_stats_for_policy(base_params, trace_files, "naive", metric)
    return statistics.median(naive_stats)


def extract_job_id(tinker_path: str) -> str:
    """Extract job ID from a Tinker path like 'tinker://UUID:train:0/weights/...'."""
    if tinker_path.startswith("tinker://"):
        path_part = tinker_path[9:]
        job_id = path_part.split(":")[0]
        return job_id
    return "unknown"


def main(config: Config):
    """Main training loop."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    assert "TINKER_API_KEY" in os.environ, "TINKER_API_KEY not set in environment"

    logger.info(f"Metric to optimize: {config.metric}")
    logger.info("Generating training traces...")

    # Generate diverse training traces
    trace_files = generate_training_traces(config)
    logger.info(f"Generated {len(trace_files)} trace files")

    # Get base simulation parameters
    base_params = get_base_params(config)

    # Compute baseline metric for reward normalization
    logger.info("Computing baseline metric with naive scheduler...")
    baseline_metric = compute_baseline_metric(base_params, trace_files, config.metric)
    logger.info(f"Baseline {config.metric}: {baseline_metric:.2f}")

    # Build system context for prompts
    system_context = build_system_context()

    # Initialize Tinker client
    service_client = tinker.ServiceClient(base_url=config.base_url)

    # Create training client
    logger.info(f"Creating LoRA training client for {config.model_name}")
    training_client = service_client.create_lora_training_client(
        base_model=config.model_name,
        rank=config.lora_rank,
    )

    # Get job ID and create log directory
    initial_state_path = training_client.save_state(name="init").result().path
    job_id = extract_job_id(initial_state_path)
    logger.info(f"Tinker job ID: {job_id}")

    job_log_path = Path(config.log_path) / job_id
    job_log_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving logs to: {job_log_path}")

    # Get tokenizer
    logger.info("Getting tokenizer from training client...")
    tokenizer = training_client.get_tokenizer()

    # Sampling parameters
    sampling_params = types.SamplingParams(
        max_tokens=config.max_tokens,
        temperature=0.7,
    )

    # Optimizer parameters
    adam_params = types.AdamParams(
        learning_rate=config.learning_rate,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
    )

    # We generate policies on demand (no pre-made dataset)
    # Each batch generates new policies and evaluates them
    n_batches = 100  # Can be configured

    logger.info(
        f"Training for {n_batches * config.num_epochs} batches ({config.num_epochs} epochs)"
    )
    logger.info(f"Batch size: {config.batch_size}, Group size: {config.group_size}")

    global_step = 0
    metrics_history = []
    samples_log_path = job_log_path / "generated_policies.jsonl"
    policy_counter = 0

    for epoch in range(config.num_epochs):
        logger.info(f"=== Epoch {epoch + 1}/{config.num_epochs} ===")

        for batch_idx in range(n_batches):
            t_start = time.time()

            # Save checkpoint periodically
            if (
                config.save_every > 0
                and global_step % config.save_every == 0
                and global_step > 0
            ):
                logger.info(f"Saving checkpoint at step {global_step}")
                state_path = (
                    training_client.save_state(name=f"step_{global_step:06d}")
                    .result()
                    .path
                )
                logger.info(f"Checkpoint saved to {state_path}")

            # Get sampling client with current weights
            sampling_client = training_client.save_weights_and_get_sampling_client(
                name=f"step_{global_step:06d}"
            )

            training_datums: list[types.Datum] = []
            batch_rewards: list[float] = []
            batch_samples: list[dict] = []

            # Generate batch_size examples
            for _ in range(config.batch_size):
                # Sample group_size completions, each with a unique key
                group_rewards: list[float] = []
                group_tokens: list[list[int]] = []
                group_logprobs: list[list[float]] = []
                group_ob_lens: list[int] = []

                for sample_idx in range(config.group_size):
                    # Each sample gets a unique key to avoid overwrites
                    policy_counter += 1
                    sample_key = f"rl_policy_{job_id[:8]}_{policy_counter}"

                    # Build prompt with this sample's unique key
                    prompt_text = build_prompt(sample_key, config.metric, system_context)
                    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
                    model_input = types.ModelInput.from_ints(tokens=prompt_tokens)

                    sample_future = sampling_client.sample(
                        prompt=model_input,
                        num_samples=1,
                        sampling_params=sampling_params,
                    )
                    sample_result = sample_future.result()
                    sampled_tokens = sample_result.sequences[0].tokens
                    sampled_logprobs = sample_result.sequences[0].logprobs
                    assert sampled_logprobs is not None

                    all_tokens = prompt_tokens + sampled_tokens
                    group_tokens.append(all_tokens)
                    group_ob_lens.append(len(prompt_tokens) - 1)
                    group_logprobs.append(sampled_logprobs)

                    # Decode and evaluate
                    generated_text = tokenizer.decode(sampled_tokens)
                    policy_code = parse_policy_code(generated_text)

                    if policy_code is None:
                        reward = 0.0
                        error_msg = "Failed to parse policy code"
                        metric_values = None
                    else:
                        # Clean the code
                        policy_code = clean_generated_code(policy_code)

                        # Check if model used the exact key we requested
                        key_match = re.search(
                            r'@register_scheduler\(key=["\']([^"\']+)["\']\)',
                            policy_code,
                        )
                        if key_match:
                            actual_key = key_match.group(1)
                        else:
                            actual_key = None

                        if actual_key != sample_key:
                            # Model didn't follow instructions - 0 reward
                            reward = 0.0
                            error_msg = f"Key mismatch: expected '{sample_key}', got '{actual_key}'"
                            metric_values = None
                        else:
                            # Key matches, execute and evaluate
                            reward, metric_values, error_msg = (
                                execute_policy_and_get_reward(
                                    policy_code,
                                    sample_key,
                                    base_params,
                                    trace_files,
                                    config.metric,
                                    baseline_metric,
                                )
                            )

                    group_rewards.append(reward)

                    # Store first sample for logging
                    if sample_idx == 0:
                        batch_samples.append(
                            {
                                "step": global_step,
                                "policy_key": sample_key,
                                "raw_response": generated_text[
                                    :2000
                                ],  # Truncate for logging
                                "policy_code": policy_code[:2000]
                                if policy_code
                                else None,
                                "reward": reward,
                                "metric_values": metric_values,
                                "error": error_msg,
                                "valid": policy_code is not None and error_msg is None,
                            }
                        )

                # Compute advantages (reward - mean)
                mean_reward = (
                    sum(group_rewards) / len(group_rewards) if group_rewards else 0.0
                )
                advantages = [r - mean_reward for r in group_rewards]
                batch_rewards.append(mean_reward)

                # Skip if all advantages are zero
                if all(a == 0.0 for a in advantages):
                    continue

                # Create training datums
                for tokens, logprob, advantage, ob_len in zip(
                    group_tokens, group_logprobs, advantages, group_ob_lens
                ):
                    input_tokens = [int(t) for t in tokens[:-1]]
                    target_tokens = tokens[1:]
                    all_logprobs = [0.0] * ob_len + list(logprob)
                    all_advantages = [0.0] * ob_len + [advantage] * (
                        len(input_tokens) - ob_len
                    )

                    # Ensure lengths match
                    min_len = min(
                        len(input_tokens),
                        len(target_tokens),
                        len(all_logprobs),
                        len(all_advantages),
                    )
                    input_tokens = input_tokens[:min_len]
                    target_tokens = target_tokens[:min_len]
                    all_logprobs = all_logprobs[:min_len]
                    all_advantages = all_advantages[:min_len]

                    datum = types.Datum(
                        model_input=types.ModelInput.from_ints(tokens=input_tokens),
                        loss_fn_inputs={
                            "target_tokens": TensorData.from_torch(
                                torch.tensor(target_tokens)
                            ),
                            "logprobs": TensorData.from_torch(
                                torch.tensor(all_logprobs, dtype=torch.float32)
                            ),
                            "advantages": TensorData.from_torch(
                                torch.tensor(all_advantages, dtype=torch.float32)
                            ),
                        },
                    )
                    training_datums.append(datum)

            # Perform training step
            if training_datums:
                fwd_bwd_future = training_client.forward_backward(
                    training_datums, loss_fn="importance_sampling"
                )
                optim_step_future = training_client.optim_step(adam_params)
                # Block until both complete
                fwd_bwd_future.result()
                optim_step_future.result()

            # Compute metrics
            elapsed = time.time() - t_start
            mean_batch_reward = (
                sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0.0
            )
            success_rate = (
                sum(1 for s in batch_samples if s["valid"]) / len(batch_samples)
                if batch_samples
                else 0.0
            )

            metrics = {
                "step": global_step,
                "epoch": epoch + 1,
                "batch": batch_idx,
                "reward": mean_batch_reward,
                "success_rate": success_rate,
                "time": elapsed,
                "num_datums": len(training_datums),
            }
            metrics_history.append(metrics)

            logger.info(
                f"Step {global_step}: reward={mean_batch_reward:.3f}, "
                f"success={success_rate:.1%}, time={elapsed:.1f}s, "
                f"datums={len(training_datums)}"
            )

            # Save samples to file
            with open(samples_log_path, "a") as f:
                for sample in batch_samples:
                    f.write(json.dumps(sample) + "\n")

            # Print one example
            if batch_samples:
                example = batch_samples[0]
                logger.info(f"--- Example from step {global_step} ---")
                logger.info(f"Policy key: {example['policy_key']}")
                logger.info(f"Valid: {example['valid']}")
                logger.info(f"Reward: {example['reward']:.3f}")
                if example["metric_values"]:
                    logger.info(f"Metric values: {example['metric_values']}")
                if example["error"]:
                    logger.info(f"Error: {example['error']}")
                logger.info("--- End example ---")

            global_step += 1

    # Save final checkpoint
    logger.info("Saving final checkpoint...")
    final_path = training_client.save_state(name="final").result().path
    logger.info(f"Final checkpoint saved to {final_path}")

    # Save metrics history
    metrics_path = job_log_path / "metrics.jsonl"
    with open(metrics_path, "w") as f:
        for m in metrics_history:
            f.write(json.dumps(m) + "\n")
    logger.info(f"Metrics saved to {metrics_path}")
    logger.info(f"Generated policies saved to {samples_log_path}")

    # Save job info
    job_info_path = job_log_path / "job_info.json"
    job_info = {
        "job_id": job_id,
        "model_name": config.model_name,
        "metric": config.metric,
        "baseline_metric": baseline_metric,
        "num_traces": len(trace_files),
        "batch_size": config.batch_size,
        "group_size": config.group_size,
        "learning_rate": config.learning_rate,
        "lora_rank": config.lora_rank,
        "num_epochs": config.num_epochs,
        "final_checkpoint": final_path,
    }
    with open(job_info_path, "w") as f:
        json.dump(job_info, f, indent=2)
    logger.info(f"Job info saved to {job_info_path}")

    logger.info("Training completed!")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
