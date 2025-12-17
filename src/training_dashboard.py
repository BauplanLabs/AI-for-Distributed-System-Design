"""
Streamlit dashboard for visualizing FaaS scheduler policy training progress.

Run with: uv run streamlit run src/training_dashboard.py
"""

import json
import random
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Paths to log files
LOG_DIR = Path(__file__).parent / "training_logs"
POLICIES_DIR = Path(__file__).parent / "policies"


def get_available_jobs() -> list[str]:
    """Get list of available job IDs (subdirectories in training_logs)."""
    if not LOG_DIR.exists():
        return []

    jobs = []
    for item in LOG_DIR.iterdir():
        if item.is_dir():
            # Check if it has generated_policies file
            if (item / "generated_policies.jsonl").exists():
                jobs.append(item.name)

    # Sort by modification time (most recent first)
    jobs.sort(key=lambda x: (LOG_DIR / x).stat().st_mtime, reverse=True)
    return jobs


def load_job_info(job_id: str) -> dict:
    """Load job info if available."""
    job_info_path = LOG_DIR / job_id / "job_info.json"
    if job_info_path.exists():
        with open(job_info_path) as f:
            return json.load(f)
    return {}


def load_policies(job_id: str) -> list[dict]:
    """Load generated policies from JSONL file for a specific job."""
    policies_path = LOG_DIR / job_id / "generated_policies.jsonl"
    if not policies_path.exists():
        return []

    policies = []
    with open(policies_path) as f:
        for line in f:
            if line.strip():
                policies.append(json.loads(line))

    return policies


def load_saved_policies() -> list[dict]:
    """Load all saved policy files from the policies directory."""
    if not POLICIES_DIR.exists():
        return []

    policies = []
    for file in sorted(POLICIES_DIR.glob("*.py")):
        with open(file) as f:
            content = f.read()

        # Extract metadata from docstring
        policy_info = {
            "filename": file.name,
            "path": str(file),
            "code": content,
        }

        # Try to parse iteration from filename (e.g., "clear_slapstick_iter1_20251126.py")
        parts = file.stem.split("_iter")
        if len(parts) > 1:
            iter_part = parts[1].split("_")[0]
            try:
                policy_info["iteration"] = int(iter_part)
            except ValueError:
                policy_info["iteration"] = 0
        else:
            policy_info["iteration"] = 0

        policies.append(policy_info)

    return policies


def compute_metrics_from_policies(policies: list[dict]) -> pd.DataFrame:
    """Compute aggregate metrics from policy generation data."""
    if not policies:
        return pd.DataFrame()

    metrics = []
    for i, p in enumerate(policies):
        metrics.append({
            "step": p.get("step", i),
            "reward": p.get("reward", 0.0),
            "valid": p.get("valid", False),
            "has_error": p.get("error") is not None,
        })

    df = pd.DataFrame(metrics)

    # Compute rolling stats if we have enough data
    if len(df) > 1:
        window = min(5, len(df))
        df["reward_rolling_avg"] = df["reward"].rolling(window=window, min_periods=1).mean()
        df["valid_rate_rolling"] = df["valid"].rolling(window=window, min_periods=1).mean()

    return df


def main():
    st.set_page_config(
        page_title="FaaS Policy Training Dashboard",
        page_icon="🚀",
        layout="wide",
    )

    st.title("🚀 FaaS Scheduler Policy Training Dashboard")

    # Get available jobs
    available_jobs = get_available_jobs()

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 Training Progress", "🔍 Policy Explorer", "📁 Saved Policies"])

    if not available_jobs:
        with tab1:
            st.warning("No training jobs found. Run training first to generate logs.")
            st.info(f"Expected log directory: `{LOG_DIR}`")
        with tab2:
            st.info("No training jobs available.")
    else:
        # Job selector in sidebar
        st.sidebar.header("Select Training Job")

        # Create display labels with job info
        job_labels = {}
        for job_id in available_jobs:
            job_info = load_job_info(job_id)
            if job_info:
                model = job_info.get("model_name", "unknown").split("/")[-1]
                metric = job_info.get("metric", "?")
                label = f"{job_id[:8]}... ({model}, {metric})"
            else:
                label = f"{job_id[:8]}..."
            job_labels[job_id] = label

        selected_job = st.sidebar.selectbox(
            "Training Job:",
            available_jobs,
            format_func=lambda x: job_labels.get(x, x),
        )

        # Show job info in sidebar
        job_info = load_job_info(selected_job)
        if job_info:
            st.sidebar.divider()
            st.sidebar.subheader("Job Details")
            st.sidebar.markdown(f"**Model:** `{job_info.get('model_name', 'N/A')}`")
            st.sidebar.markdown(f"**Metric:** {job_info.get('metric', 'N/A')}")
            st.sidebar.markdown(f"**Baseline:** {job_info.get('baseline_metric', 'N/A'):.2f}" if job_info.get('baseline_metric') else "**Baseline:** N/A")
            st.sidebar.markdown(f"**Num Traces:** {job_info.get('num_traces', 'N/A')}")
            st.sidebar.markdown(f"**Batch Size:** {job_info.get('batch_size', 'N/A')}")
            st.sidebar.markdown(f"**Group Size:** {job_info.get('group_size', 'N/A')}")
            st.sidebar.markdown(f"**Epochs:** {job_info.get('num_epochs', 'N/A')}")
            st.sidebar.markdown(f"**Learning Rate:** {job_info.get('learning_rate', 'N/A')}")
            st.sidebar.markdown(f"**LoRA Rank:** {job_info.get('lora_rank', 'N/A')}")

            if job_info.get("final_checkpoint"):
                st.sidebar.divider()
                st.sidebar.markdown("**Final Checkpoint:**")
                st.sidebar.code(job_info["final_checkpoint"], language=None)

        # Load data for selected job
        policies = load_policies(selected_job)

        if not policies:
            with tab1:
                st.warning(f"No policy data found for job {selected_job}")
            with tab2:
                st.warning(f"No policies found for job {selected_job}")
        else:
            metrics_df = compute_metrics_from_policies(policies)

            # Tab 1: Training Progress Charts
            with tab1:
                st.header("Training Metrics Over Time")

                col1, col2 = st.columns(2)

                with col1:
                    # Reward over steps
                    fig_reward = px.line(
                        metrics_df,
                        x="step",
                        y="reward",
                        title="Reward Over Training Steps",
                        labels={"step": "Training Step", "reward": "Reward"},
                    )
                    fig_reward.update_traces(line=dict(color="#636EFA", width=2))

                    # Add rolling average if available
                    if "reward_rolling_avg" in metrics_df.columns:
                        fig_reward.add_scatter(
                            x=metrics_df["step"],
                            y=metrics_df["reward_rolling_avg"],
                            mode="lines",
                            name="Rolling Avg",
                            line=dict(color="#00CC96", width=2, dash="dash"),
                        )

                    st.plotly_chart(fig_reward, use_container_width=True)

                with col2:
                    # Valid policy rate over steps
                    if "valid_rate_rolling" in metrics_df.columns:
                        fig_valid = px.line(
                            metrics_df,
                            x="step",
                            y="valid_rate_rolling",
                            title="Policy Validity Rate (Rolling Avg)",
                            labels={"step": "Training Step", "valid_rate_rolling": "Valid Rate"},
                        )
                        fig_valid.update_traces(line=dict(color="#00CC96", width=2))
                        fig_valid.update_layout(yaxis_tickformat=".0%")
                        st.plotly_chart(fig_valid, use_container_width=True)
                    else:
                        # Simple bar chart of valid vs invalid
                        valid_count = sum(1 for p in policies if p.get("valid", False))
                        invalid_count = len(policies) - valid_count
                        fig_pie = px.pie(
                            values=[valid_count, invalid_count],
                            names=["Valid", "Invalid"],
                            title="Policy Validity Distribution",
                            color_discrete_sequence=["#00CC96", "#EF553B"],
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)

                # Summary stats
                st.subheader("Summary Statistics")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Total Policies",
                        len(policies),
                    )
                with col2:
                    valid_count = sum(1 for p in policies if p.get("valid", False))
                    st.metric(
                        "Valid Policies",
                        f"{valid_count} ({100 * valid_count / len(policies):.1f}%)",
                    )
                with col3:
                    st.metric(
                        "Best Reward",
                        f"{metrics_df['reward'].max():.3f}",
                    )
                with col4:
                    rewarded = [p for p in policies if p.get("reward", 0) > 0]
                    st.metric(
                        "Avg Reward (valid)",
                        f"{sum(p['reward'] for p in rewarded) / len(rewarded):.3f}"
                        if rewarded else "N/A",
                    )

                # Reward distribution
                st.subheader("Reward Distribution")
                valid_rewards = [p.get("reward", 0) for p in policies if p.get("valid", False)]
                if valid_rewards:
                    fig_hist = px.histogram(
                        x=valid_rewards,
                        nbins=20,
                        title="Distribution of Rewards (Valid Policies Only)",
                        labels={"x": "Reward", "y": "Count"},
                    )
                    fig_hist.update_traces(marker_color="#636EFA")
                    st.plotly_chart(fig_hist, use_container_width=True)
                else:
                    st.info("No valid policies to show reward distribution.")

                # Error analysis
                st.subheader("Error Analysis")
                errors = [p.get("error") for p in policies if p.get("error")]
                if errors:
                    # Count error types
                    error_counts = {}
                    for err in errors:
                        # Truncate long errors for grouping
                        err_key = err[:80] if len(err) > 80 else err
                        error_counts[err_key] = error_counts.get(err_key, 0) + 1

                    error_df = pd.DataFrame([
                        {"Error": k, "Count": v}
                        for k, v in sorted(error_counts.items(), key=lambda x: -x[1])
                    ])
                    st.dataframe(error_df, use_container_width=True)
                else:
                    st.success("No errors recorded!")

            # Tab 2: Policy Explorer
            with tab2:
                st.header("Generated Policies Explorer")

                # Stats
                total_policies = len(policies)
                valid_policies = sum(1 for p in policies if p.get("valid", False))
                rewarded = sum(1 for p in policies if p.get("reward", 0) > 0)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Samples", total_policies)
                with col2:
                    st.metric(
                        "Valid Policies",
                        f"{valid_policies} ({100 * valid_policies / total_policies:.1f}%)",
                    )
                with col3:
                    st.metric(
                        "Rewarded (>0)",
                        f"{rewarded} ({100 * rewarded / total_policies:.1f}%)",
                    )

                st.divider()

                # "I Feel Lucky" button and filters
                col1, col2 = st.columns([1, 3])
                with col1:
                    if st.button("🎲 Random Policy!", use_container_width=True):
                        st.session_state.selected_policy_idx = random.randint(0, len(policies) - 1)

                with col2:
                    filter_option = st.selectbox(
                        "Filter policies:",
                        ["All", "Valid & Rewarded", "Valid but Zero Reward", "Invalid/Error"],
                    )

                # Filter policies based on selection
                if filter_option == "Valid & Rewarded":
                    filtered_policies = [p for p in policies if p.get("reward", 0) > 0]
                elif filter_option == "Valid but Zero Reward":
                    filtered_policies = [
                        p for p in policies
                        if p.get("valid", False) and p.get("reward", 0) == 0
                    ]
                elif filter_option == "Invalid/Error":
                    filtered_policies = [p for p in policies if not p.get("valid", False)]
                else:
                    filtered_policies = policies

                if filtered_policies:
                    # Sort options
                    sort_option = st.selectbox(
                        "Sort by:",
                        ["Step (ascending)", "Step (descending)", "Reward (highest first)", "Reward (lowest first)"],
                    )

                    if sort_option == "Step (descending)":
                        filtered_policies = sorted(filtered_policies, key=lambda p: p.get("step", 0), reverse=True)
                    elif sort_option == "Reward (highest first)":
                        filtered_policies = sorted(filtered_policies, key=lambda p: p.get("reward", 0), reverse=True)
                    elif sort_option == "Reward (lowest first)":
                        filtered_policies = sorted(filtered_policies, key=lambda p: p.get("reward", 0))

                    # Show slider to browse policies
                    idx = st.slider(
                        "Browse policies:",
                        0,
                        len(filtered_policies) - 1,
                        st.session_state.get("selected_policy_idx", 0) % len(filtered_policies),
                        key="policy_slider",
                    )
                    display_policy(filtered_policies[idx])
                else:
                    st.info("No policies match the current filter.")

    # Tab 3: Saved Policies (always available)
    with tab3:
        st.header("Saved Policy Files")
        saved_policies = load_saved_policies()

        if not saved_policies:
            st.info(f"No saved policy files found in `{POLICIES_DIR}`")
        else:
            st.markdown(f"Found **{len(saved_policies)}** saved policy files.")

            # Group by iteration
            col1, col2 = st.columns([1, 3])
            with col1:
                selected_idx = st.selectbox(
                    "Select policy:",
                    range(len(saved_policies)),
                    format_func=lambda i: saved_policies[i]["filename"],
                )

            with col2:
                if st.button("📋 Copy to Clipboard", key="copy_saved"):
                    st.info("Use the code block below to copy the policy code.")

            if selected_idx is not None:
                policy = saved_policies[selected_idx]
                st.subheader(f"Policy: {policy['filename']}")
                st.code(policy["code"], language="python", line_numbers=True)


def display_policy(policy: dict):
    """Display a single policy with nice formatting."""
    st.divider()

    # Header with validity status
    is_valid = policy.get("valid", False)
    has_reward = policy.get("reward", 0) > 0
    error = policy.get("error")

    if has_reward:
        status = "✅ Valid & Rewarded"
        status_color = "green"
    elif is_valid:
        status = "⚠️ Valid, Zero Reward"
        status_color = "orange"
    else:
        status = "❌ Invalid/Error"
        status_color = "red"

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.subheader(f"Step {policy.get('step', 'N/A')}")
    with col2:
        st.markdown(f"**Status:** :{status_color}[{status}]")
    with col3:
        st.metric("Reward", f"{policy.get('reward', 0):.3f}")

    # Policy key
    st.markdown(f"**Policy Key:** `{policy.get('policy_key', 'N/A')}`")

    # Error message if any
    if error:
        st.error(f"**Error:** {error}")

    # Metric values
    metric_values = policy.get("metric_values")
    if metric_values:
        st.markdown("**Metric Values:**")
        st.json(metric_values)

    # Policy Code
    st.markdown("### Policy Code")
    policy_code = policy.get("policy_code")
    if policy_code:
        st.code(policy_code, language="python", line_numbers=True)
    else:
        st.warning("No policy code extracted")

    # Raw response (collapsed)
    with st.expander("View Raw LLM Response"):
        raw = policy.get("raw_response", "N/A")
        st.text(raw if raw else "N/A")


if __name__ == "__main__":
    main()
