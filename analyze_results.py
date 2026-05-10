"""
Analyze and compare baseline vs masked-CoT GRPO runs.

Usage:
    python analyze_results.py --baseline_dir outputs/baseline_* --masked_dir outputs/masked_cot_*
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        print(f"  Warning: {path} not found")
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def plot_cot_evolution(baseline_dir: str, masked_dir: str, output_path: str):
    """Plot CoT statistics over training for both runs."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("GRPO CoT Masking Experiment: Baseline vs Masked CoT", fontsize=14)

    for label, run_dir, color in [
        ("Baseline (full loss)", baseline_dir, "tab:blue"),
        ("Masked CoT (answer-only loss)", masked_dir, "tab:orange"),
    ]:
        log_dir = os.path.join(run_dir, "cot_logs")
        samples = load_jsonl(os.path.join(log_dir, "completion_samples.jsonl"))
        metrics = load_jsonl(os.path.join(log_dir, "metrics.jsonl"))

        if not samples:
            print(f"  No completion samples for {label}")
            continue

        # Group by step
        steps = sorted(set(s["step"] for s in samples))
        mean_cot_len = []
        mean_answer_len = []
        think_tag_rate = []
        for step in steps:
            step_samples = [s for s in samples if s["step"] == step]
            mean_cot_len.append(np.mean([s["cot_length_chars"] for s in step_samples]))
            mean_answer_len.append(np.mean([s["answer_length_chars"] for s in step_samples]))
            think_tag_rate.append(np.mean([s["has_think_tags"] for s in step_samples]))

        # Plot 1: CoT length over time
        axes[0, 0].plot(steps, mean_cot_len, label=label, color=color, marker="o", markersize=3)
        axes[0, 0].set_title("Mean CoT Length (chars)")
        axes[0, 0].set_xlabel("Training Step")
        axes[0, 0].set_ylabel("Characters")
        axes[0, 0].legend()

        # Plot 2: Answer length over time
        axes[0, 1].plot(steps, mean_answer_len, label=label, color=color, marker="o", markersize=3)
        axes[0, 1].set_title("Mean Answer Length (chars)")
        axes[0, 1].set_xlabel("Training Step")
        axes[0, 1].set_ylabel("Characters")
        axes[0, 1].legend()

        # Plot 3: Think tag presence rate
        axes[1, 0].plot(steps, think_tag_rate, label=label, color=color, marker="o", markersize=3)
        axes[1, 0].set_title("</think> Tag Presence Rate")
        axes[1, 0].set_xlabel("Training Step")
        axes[1, 0].set_ylabel("Fraction")
        axes[1, 0].set_ylim(-0.05, 1.05)
        axes[1, 0].legend()

        # Plot 4: Reward from metrics (if available)
        if metrics:
            reward_metrics = [m for m in metrics if "reward" in m or "reward/mean" in m]
            if reward_metrics:
                rsteps = [m["step"] for m in reward_metrics]
                rvals = [m.get("reward", m.get("reward/mean", 0)) for m in reward_metrics]
                axes[1, 1].plot(rsteps, rvals, label=label, color=color, marker="o", markersize=3)

    axes[1, 1].set_title("Mean Reward")
    axes[1, 1].set_xlabel("Training Step")
    axes[1, 1].set_ylabel("Reward")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


def print_example_cots(run_dir: str, label: str, steps: list[int] | None = None):
    """Print example CoTs at specific training steps."""
    log_dir = os.path.join(run_dir, "cot_logs")
    samples = load_jsonl(os.path.join(log_dir, "completion_samples.jsonl"))
    if not samples:
        print(f"  No samples for {label}")
        return

    all_steps = sorted(set(s["step"] for s in samples))
    if steps is None:
        # Show first, middle, and last
        if len(all_steps) >= 3:
            steps = [all_steps[0], all_steps[len(all_steps) // 2], all_steps[-1]]
        else:
            steps = all_steps

    print(f"\n{'=' * 70}")
    print(f"  {label} — Example Completions")
    print(f"{'=' * 70}")

    for step in steps:
        step_samples = [s for s in samples if s["step"] == step]
        if not step_samples:
            continue
        s = step_samples[0]  # Show first example
        print(f"\n--- Step {step} ---")
        print(f"  Prompt: {s['prompt'][:120]}...")
        print(f"  CoT length: {s['cot_length_chars']} chars | Answer length: {s['answer_length_chars']} chars")
        print(f"  Has </think>: {s['has_think_tags']}")
        print(f"  Completion:\n    {s['completion'][:500]}")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_dir", type=str, required=True)
    parser.add_argument("--masked_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="cot_comparison.png")
    args = parser.parse_args()

    plot_cot_evolution(args.baseline_dir, args.masked_dir, args.output)
    print_example_cots(args.baseline_dir, "Baseline")
    print_example_cots(args.masked_dir, "Masked CoT")


if __name__ == "__main__":
    main()
