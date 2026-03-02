"""
Main evaluation script for CodeSteer reproduction.

Usage:
  python run_eval.py --tasks game24 sudoku --n_samples 10 --output results/run.json
  python run_eval.py --all_tasks --n_samples 100 --output results/full.json
  python run_eval.py --tasks game24 --baselines --n_samples 20 --output results/comparison.json
"""

import argparse
import json
import os
import sys
import time
import yaml
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from codesteer.steer_llm import CodeSteerLLM
from codesteer.task_llm import TaskLLM
from codesteer.pipeline import CodeSteerPipeline, BaselinePipeline
from codesteer.tasks import TASK_REGISTRY, ALL_TASKS


def load_config(path: str = "config.yaml") -> dict:
    if not os.path.exists(path):
        print(f"Config file '{path}' not found. Copy config.example.yaml to config.yaml and fill in your API keys.")
        sys.exit(1)
    with open(path) as f:
        return yaml.safe_load(f)


def run_task(pipeline, task_cls, n_samples: int, complexity: int = 1) -> list[dict]:
    task = task_cls()
    results = []
    for i in tqdm(range(n_samples), desc=task.name, leave=False):
        sample = task.generate_question(complexity=complexity, seed=i * 42 + 7)
        try:
            result = pipeline.run(
                question=sample["question"],
                correct_answer=sample["answer"],
                task_name=task.name,
            )
            # Use task's own checker if available
            success = task.check_answer(
                sample["question"],
                result.final_answer,
                metadata=sample.get("metadata"),
            )
            results.append({
                "task": task.name,
                "question": sample["question"],
                "correct_answer": sample["answer"],
                "predicted_answer": result.final_answer,
                "success": success,
                "n_turns": result.n_turns,
                "method": result.method,
            })
        except Exception as e:
            print(f"\n  Error on sample {i}: {e}")
            results.append({
                "task": task.name,
                "question": sample["question"],
                "correct_answer": sample["answer"],
                "predicted_answer": "",
                "success": False,
                "n_turns": 0,
                "method": "error",
                "error": str(e),
            })
        time.sleep(0.5)  # rate limiting

    return results


def compute_stats(results: list[dict]) -> dict:
    """Compute per-task and overall success rates."""
    by_task = {}
    for r in results:
        task = r["task"]
        method = r.get("method", "codesteer")
        key = f"{task}_{method}"
        if key not in by_task:
            by_task[key] = {"task": task, "method": method, "successes": 0, "total": 0}
        by_task[key]["total"] += 1
        by_task[key]["successes"] += int(r["success"])

    stats = []
    for key, v in by_task.items():
        rate = v["successes"] / v["total"] * 100 if v["total"] > 0 else 0
        stats.append({
            "task": v["task"],
            "method": v["method"],
            "success_rate": round(rate, 1),
            "successes": v["successes"],
            "total": v["total"],
        })

    return {"per_task": stats, "total_samples": len(results)}


def main():
    parser = argparse.ArgumentParser(description="CodeSteer Reproduction Evaluation")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--tasks", nargs="+", choices=ALL_TASKS, help="Tasks to evaluate")
    parser.add_argument("--all_tasks", action="store_true", help="Run all tasks")
    parser.add_argument("--n_samples", type=int, default=10, help="Samples per task")
    parser.add_argument("--complexity", type=int, default=1, choices=[1, 2, 3], help="Task complexity (1=easy, 3=hard)")
    parser.add_argument("--output", default="results/run.json", help="Output JSON path")
    parser.add_argument("--baselines", action="store_true", help="Also run baseline methods for comparison")
    parser.add_argument("--skip_codesteer", action="store_true", help="Skip CodeSteer (baselines only)")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Select tasks
    task_names = ALL_TASKS if args.all_tasks else (args.tasks or ["game24", "number_multiply"])
    print(f"\n=== CodeSteer Reproduction ===")
    print(f"Tasks: {task_names}")
    print(f"Samples per task: {args.n_samples}")
    print(f"Complexity: {args.complexity}")
    print()

    # Initialize models
    task_llm = TaskLLM(config["task_llm"])
    print(f"TaskLLM: {config['task_llm']['provider']} / {config['task_llm']['model']}")

    all_results = []

    # CodeSteer pipeline
    if not args.skip_codesteer:
        print("\nLoading CodeSteerLLM...")
        steer_llm = CodeSteerLLM(config["codesteer_llm"])
        pipeline = CodeSteerPipeline(steer_llm, task_llm, config.get("eval", {}))

        for task_name in task_names:
            print(f"\n[CodeSteer] Running task: {task_name}")
            task_cls = TASK_REGISTRY[task_name]
            results = run_task(pipeline, task_cls, args.n_samples, args.complexity)
            for r in results:
                r["method"] = "codesteer"
            all_results.extend(results)

    # Baseline pipelines
    if args.baselines:
        for method in ["only_question", "text_only", "code_only"]:
            baseline = BaselinePipeline(task_llm, method, config.get("eval", {}))
            for task_name in task_names:
                print(f"\n[{method}] Running task: {task_name}")
                task_cls = TASK_REGISTRY[task_name]
                results = run_task(baseline, task_cls, args.n_samples, args.complexity)
                for r in results:
                    r["method"] = method
                all_results.extend(results)

    # Compute stats
    stats = compute_stats(all_results)

    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    output_data = {
        "config": {
            "tasks": task_names,
            "n_samples": args.n_samples,
            "complexity": args.complexity,
            "task_llm": config["task_llm"].get("model"),
        },
        "stats": stats,
        "results": all_results,
    }
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n\n=== Results Summary ===")
    print(f"{'Task':<25} {'Method':<20} {'Success Rate':>12}")
    print("-" * 60)
    for s in sorted(stats["per_task"], key=lambda x: (x["task"], x["method"])):
        print(f"{s['task']:<25} {s['method']:<20} {s['success_rate']:>11.1f}%")

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
