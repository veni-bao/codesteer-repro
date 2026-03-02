"""
Plot evaluation results as bar charts and comparison figures.

Usage:
  python plot_results.py --input results/run.json --output figures/
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


METHOD_COLORS = {
    "codesteer":    "#2196F3",   # blue
    "only_question":"#9E9E9E",   # grey
    "text_only":    "#FF9800",   # orange
    "code_only":    "#4CAF50",   # green
}

METHOD_LABELS = {
    "codesteer":    "CodeSteer (ours)",
    "only_question":"Only Question",
    "text_only":    "All Text + CoT",
    "code_only":    "All Code + CoT",
}


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_per_task_comparison(stats: list[dict], output_dir: str):
    """Bar chart: success rate per task, grouped by method."""
    tasks = sorted(set(s["task"] for s in stats))
    methods = sorted(set(s["method"] for s in stats))

    # Build lookup
    lookup = {}
    for s in stats:
        lookup[(s["task"], s["method"])] = s["success_rate"]

    x = np.arange(len(tasks))
    width = 0.8 / max(len(methods), 1)

    fig, ax = plt.subplots(figsize=(max(10, len(tasks) * 1.5), 6))

    for i, method in enumerate(methods):
        values = [lookup.get((task, method), 0) for task in tasks]
        offset = (i - len(methods) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width * 0.9,
                      label=METHOD_LABELS.get(method, method),
                      color=METHOD_COLORS.get(method, "#607D8B"),
                      alpha=0.85)
        # Value labels
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{val:.0f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "\n") for t in tasks], fontsize=9)
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("CodeSteer vs. Baselines — SymBench Tasks", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "per_task_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_overall_bar(stats: list[dict], output_dir: str):
    """Overall average success rate per method."""
    from collections import defaultdict
    method_rates = defaultdict(list)
    for s in stats:
        method_rates[s["method"]].append(s["success_rate"])

    methods = sorted(method_rates.keys())
    averages = [np.mean(method_rates[m]) for m in methods]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [METHOD_COLORS.get(m, "#607D8B") for m in methods]
    labels = [METHOD_LABELS.get(m, m) for m in methods]
    bars = ax.bar(labels, averages, color=colors, alpha=0.85, width=0.5)

    for bar, val in zip(bars, averages):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontweight="bold")

    ax.set_ylabel("Average Success Rate (%)")
    ax.set_title("Overall Performance — CodeSteer Reproduction", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "overall_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_paper_reference(output_dir: str):
    """
    Plot the paper's reported numbers (Table 1) for reference,
    showing key results even without running experiments.
    """
    tasks = ["Game 24", "Sudoku", "Path Plan", "BoxLift",
             "Logical\nDeduction", "Navigation", "Number\nMultiply",
             "Statistical\nCounting"]
    paper_data = {
        "GPT-4o\n(baseline)":  [17, 0, 65, 69, 89, 98, 11, 34],
        "o1":                  [80, 0, 74, 95, 100, 100, 43, 25],
        "DeepSeek R1":         [65, 0, 60, 92, 98, 100, 46, 72],
        "CodeSteer\n(GPT-4o)": [93, 100, 75, 77, 92, 99, 95, 97],
    }
    colors = ["#9E9E9E", "#FF9800", "#E91E63", "#2196F3"]

    x = np.arange(len(tasks))
    width = 0.18
    fig, ax = plt.subplots(figsize=(14, 6))

    for i, (method, values) in enumerate(paper_data.items()):
        offset = (i - 1.5) * width
        ax.bar(x + offset, values, width, label=method, color=colors[i], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=9)
    ax.set_ylabel("Task Success Rate (%)")
    ax.set_title("CodeSteer Paper Results (Table 1) — Selected Tasks", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate the key insight
    ax.annotate(
        "CodeSteer makes GPT-4o\nsolve Sudoku (0% → 100%)\nand Game 24 (17% → 93%)",
        xy=(0, 93), xytext=(1.5, 108),
        arrowprops=dict(arrowstyle="->", color="#2196F3"),
        fontsize=8, color="#2196F3",
    )

    plt.tight_layout()
    out_path = os.path.join(output_dir, "paper_reference_results.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Results JSON from run_eval.py")
    parser.add_argument("--output", default="figures", help="Output directory")
    parser.add_argument("--paper_reference", action="store_true",
                        help="Also plot paper reference results (no API needed)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Always plot paper reference
    print("Generating paper reference chart...")
    plot_paper_reference(args.output)

    if args.input and os.path.exists(args.input):
        data = load_results(args.input)
        stats = data.get("stats", {}).get("per_task", [])
        if stats:
            plot_per_task_comparison(stats, args.output)
            plot_overall_bar(stats, args.output)
        else:
            print("No per-task stats found in results file.")
    else:
        print("\nNo results file provided. Only paper reference chart generated.")
        print("Run run_eval.py first, then pass --input results/run.json")


if __name__ == "__main__":
    main()
