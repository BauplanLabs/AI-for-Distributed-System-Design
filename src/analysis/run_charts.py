#!/usr/bin/env python3
"""Generate charts comparing latency improvements across models."""

import json
from pathlib import Path
import matplotlib.pyplot as plt


def load_experiment_data(experiment_runs_dir: Path) -> dict:
    """Load summary data from all experiment runs."""
    data = {}
    for summary_path in experiment_runs_dir.glob("*/summary.json"):
        with open(summary_path) as f:
            summary = json.load(f)
        model = summary["model"]
        data[model] = summary
    return data


def extract_best_latency_series(timeseries: list) -> tuple[list, list]:
    """Extract iteration numbers and best-so-far latency values."""
    iterations = []
    best_latencies = []
    current_best = float("inf")

    for entry in timeseries:
        iteration = entry["iteration"]
        metric = entry.get("median_metric")

        # Skip failed iterations or Infinity values
        if metric is None or metric == "Infinity" or metric == "-Infinity":
            # Still record the iteration with current best
            iterations.append(iteration)
            best_latencies.append(
                current_best if current_best != float("inf") else None
            )
            continue

        # Update best if this is better (lower latency is better)
        if metric < current_best:
            current_best = metric

        iterations.append(iteration)
        best_latencies.append(current_best)

    return iterations, best_latencies


def plot_latency_comparison(data: dict, output_path: Path):
    """Create a line chart comparing latency across models over iterations."""
    plt.figure(figsize=(12, 7))

    # Color scheme for models
    colors = {
        "claude-sonnet-4-5-20250929": "#2ecc71",  # Green
        "gpt-5.2-2025-12-11": "#3498db",  # Blue
        "claude-opus-4-5-20251101": "#9b59b6",  # Purple
    }

    # Friendly names for legend
    friendly_names = {
        "claude-sonnet-4-5-20250929": "Claude Sonnet 4.5",
        "gpt-5.2-2025-12-11": "GPT-5.2",
        "claude-opus-4-5-20251101": "Claude Opus 4.5",
    }

    for model, summary in data.items():
        timeseries = summary.get("iteration_timeseries", [])
        iterations, best_latencies = extract_best_latency_series(timeseries)

        # Filter out None values for plotting
        valid_points = [
            (i, lat) for i, lat in zip(iterations, best_latencies) if lat is not None
        ]
        if not valid_points:
            continue

        x, y = zip(*valid_points)
        color = colors.get(model, "#95a5a6")
        label = friendly_names.get(model, model)

        plt.plot(x, y, marker="o", markersize=3, linewidth=2, color=color, label=label)

    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Best Latency (lower is better)", fontsize=12)
    plt.title("Latency Improvement Over Iterations by Model", fontsize=14)
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale("log")  # Log scale to better show improvements
    plt.xlim(0, 30)  # Cut x-axis at iteration 30

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Chart saved to: {output_path}")
    plt.close()


def plot_latency_linear(data: dict, output_path: Path):
    """Create a linear scale chart focusing on final convergence."""
    plt.figure(figsize=(12, 7))

    colors = {
        "claude-sonnet-4-5-20250929": "#2ecc71",
        "gpt-5.2-2025-12-11": "#3498db",
        "claude-opus-4-5-20251101": "#9b59b6",
    }

    friendly_names = {
        "claude-sonnet-4-5-20250929": "Claude Sonnet 4.5",
        "gpt-5.2-2025-12-11": "GPT-5.2",
        "claude-opus-4-5-20251101": "Claude Opus 4.5",
    }

    for model, summary in data.items():
        timeseries = summary.get("iteration_timeseries", [])
        iterations, best_latencies = extract_best_latency_series(timeseries)

        valid_points = [
            (i, lat)
            for i, lat in zip(iterations, best_latencies)
            if lat is not None and lat < 1000  # Focus on converged values
        ]
        if not valid_points:
            continue

        x, y = zip(*valid_points)
        color = colors.get(model, "#95a5a6")
        label = friendly_names.get(model, model)

        plt.plot(x, y, marker="o", markersize=3, linewidth=2, color=color, label=label)

    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Best Latency (lower is better)", fontsize=12)
    plt.title("Latency Convergence (Linear Scale, < 1000)", fontsize=14)
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 30)  # Cut x-axis at iteration 30

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Chart saved to: {output_path}")
    plt.close()


def main():
    script_dir = Path(__file__).parent
    experiment_runs_dir = script_dir.parent / "experiment_runs"

    if not experiment_runs_dir.exists():
        print(f"Error: {experiment_runs_dir} does not exist")
        return

    print(f"Loading experiments from: {experiment_runs_dir}")
    data = load_experiment_data(experiment_runs_dir)
    print(f"Found {len(data)} experiments: {list(data.keys())}")

    # Generate charts
    output_dir = script_dir
    plot_latency_comparison(data, output_dir / "latency_comparison_log.png")
    plot_latency_linear(data, output_dir / "latency_comparison_linear.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
