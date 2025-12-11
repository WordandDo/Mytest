#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Science Dataset Parallel Rollout Runner (MCP Native)

This script mirrors src/run_parallel_rollout.py but specializes in
loading the Science dataset (new JSON structure with images, sub_goals, metadata)
and running the same parallel rollout + evaluation pipeline.

Key features preserved from run_parallel_rollout.py:
- Parallel worker processes
- Environment factory usage (MCP Http env by default)
- Metrics computation and score export
- Trajectory saving

Usage examples:
  python src/run_science_rollout.py \
    --data_dir src/data/Science \
    --num_rollouts 2 \
    --output_dir results_science

  # Or point to a single file
  python src/run_science_rollout.py --data_path src/data/Science/001_....json
"""

import os
import sys
import glob
from typing import List

# Ensure this 'src' directory is importable for `import benchmark`
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from benchmark import Benchmark
from run_parallel_rollout import run_parallel_rollout, ParallelRolloutConfig
from envs.factory import register_environment, is_registered, get_environment_class


class ScienceMockEnvironment:
    """
    Minimal mock environment to exercise the rollout pipeline locally.
    - No external services or SDKs
    - Returns ground-truth answer as prediction for evaluation sanity check
    """
    has_heavy_resource = False
    active_resources = []

    def __init__(self, **kwargs):
        self.config = dict(kwargs)

    # Optional hooks used by worker (no-op)
    def env_start(self):
        return None

    def env_close(self):
        return None

    def initialize_with_task_config(self, cfg):
        return None

    def run_task(self, task, agent_config, logger):
        task_id = task.get("id", "unknown")
        question = task.get("question", "")
        # For pipeline validation, echo the ground truth as the model answer
        pred_answer = task.get("answer", "")
        return {
            "task_id": task_id,
            "question": question,
            "answer": pred_answer,
            "messages": [
                {"role": "system", "content": "ScienceMockEnvironment session"},
                {"role": "assistant", "content": pred_answer},
            ],
            "success": True,
            "error": None,
        }


def run_science_rollout_sequential(config: ParallelRolloutConfig, benchmark: Benchmark):
    """Fallback sequential runner when multiprocessing is unavailable."""
    import logging
    import json as _json

    logger = logging.getLogger("science_rollout_seq")
    logger.setLevel(logging.INFO)

    EnvClass = get_environment_class(config.env_mode)
    logger.info(f"[SEQ] Using environment class: {EnvClass.__name__}")

    environment = EnvClass(parallel_degree=1, **config.env_kwargs)
    env_start = getattr(environment, "env_start", None)
    if callable(env_start):
        try:
            environment.env_start()
        except Exception:
            pass

    agent_config = dict(config.agent_config_dict)
    agent_config["output_dir"] = config.output_dir

    results = []
    for item in benchmark.get_items():
        task = {
            "id": item.id,
            "question": item.question,
            "answer": item.answer,
            "metadata": item.metadata or {},
        }
        res = environment.run_task(task, agent_config, logger)
        results.append(res)

    # Save trajectories
    os.makedirs(config.output_dir, exist_ok=True)
    traj_path = os.path.join(config.output_dir, "trajectory.jsonl")
    with open(traj_path, "w", encoding="utf-8") as f:
        for r in results:
            _json.dump(r, f, ensure_ascii=False)
            f.write("\n")
    print(f"[SEQ] Saved trajectories to {traj_path}")

    # Evaluate
    predictions = {r["task_id"]: r.get("answer", "") for r in results if r.get("success")}
    evaluation_cfg = agent_config.get("evaluation_metric", "exact_match")
    metrics = [evaluation_cfg] if isinstance(evaluation_cfg, str) else list(evaluation_cfg)

    stats = {}
    for m in metrics:
        metric_results = benchmark.evaluate(predictions, metric=m, concurrent=False)
        total = len(metric_results)
        avg = sum(r.score for r in metric_results) / total if total else 0.0
        stats[m] = {
            "total_items": total,
            "average_score": avg,
        }
    summary_path = os.path.join(config.output_dir, "evaluation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        _json.dump({"metrics": stats, "benchmark": benchmark.get_summary()}, f, ensure_ascii=False, indent=2)
    print(f"[SEQ] Saved evaluation summary to {summary_path}")


def load_science_benchmark_from_dir(data_dir: str, pattern: str = "*.json") -> Benchmark:
    """Aggregate all Science JSON files in a directory into a single Benchmark."""
    files: List[str] = sorted(glob.glob(os.path.join(data_dir, pattern)))
    agg = Benchmark(name="Science Benchmark", description=f"Aggregated from {data_dir}")
    count = 0
    for f in files:
        try:
            b = Benchmark(data_path=f)
            if b.items:
                agg.items.extend(b.items)
                count += len(b.items)
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")
            continue
    print(f"Loaded {count} item(s) from {len(files)} file(s) under {data_dir}")
    return agg


def load_science_benchmark(data_dir: str = None, data_path: str = None, pattern: str = "*.json") -> Benchmark:
    """Load benchmark either from a directory (aggregate) or a single file."""
    if data_dir:
        return load_science_benchmark_from_dir(data_dir, pattern)
    if data_path:
        return Benchmark(data_path=data_path, name="Science Benchmark")
    raise ValueError("Must provide either data_dir or data_path")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Science dataset parallel rollout (MCP Native)")
    parser.add_argument("--data_dir", type=str, default=os.path.join(SRC_DIR, "data", "Science"), help="Directory containing Science JSON files")
    parser.add_argument("--data_path", type=str, default=None, help="Optional single JSON file to run instead of a directory")
    parser.add_argument("--pattern", type=str, default="*.json", help="Glob pattern for JSON files in data_dir")

    parser.add_argument("--num_rollouts", type=int, default=5, help="Number of parallel workers")
    parser.add_argument("--env_mode", type=str, default="science_mock", help="Environment mode (science_mock or http_mcp)")
    parser.add_argument("--output_dir", type=str, default="results_science", help="Output directory")

    # MCP / Gateway config
    parser.add_argument("--mcp_server_url", type=str, default="http://localhost:8080", help="MCP Server URL")
    parser.add_argument("--resource_api_url", type=str, default="http://localhost:8000", help="Resource API URL")
    parser.add_argument("--gateway_config_path", type=str, default="gateway_config.json", help="Path to gateway config file")

    # Agent config
    parser.add_argument("--model_name", type=str, default="gpt-4.1-2025-04-14", help="Agent model name")
    parser.add_argument("--max_turns", type=int, default=15, help="Max turns per task")
    parser.add_argument("--prompt_type", type=str, default="generic",
                        choices=["generic", "no_tool", "sparse", "hybrid"],
                        help="Prompt type for RAG environment: generic, no_tool, sparse, or hybrid")
    parser.add_argument(
        "--evaluation_metric",
        type=str,
        nargs='+',
        default=["exact_match"],
        help="Evaluation metric(s) to use. Can specify multiple: --evaluation_metric exact_match f1_score"
    )

    args = parser.parse_args()

    # Load benchmark from directory or single file
    if args.data_path:
        benchmark = load_science_benchmark(data_path=args.data_path)
    else:
        benchmark = load_science_benchmark(data_dir=args.data_dir, pattern=args.pattern)

    # Register local mock environment for convenience (if not already registered)
    if not is_registered("science_mock"):
        register_environment("science_mock", ScienceMockEnvironment)

    # Prepare environment kwargs (same as run_parallel_rollout)
    env_kwargs = {
        "observation_type": "screenshot_a11y_tree",
        "mcp_server_url": args.mcp_server_url,
        "resource_api_url": args.resource_api_url,
        "gateway_config_path": args.gateway_config_path,
        "prompt_type": args.prompt_type,
    }

    config = ParallelRolloutConfig(
        num_rollouts=args.num_rollouts,
        env_mode=args.env_mode,
        output_dir=args.output_dir,
        env_kwargs=env_kwargs,
        agent_config_dict={
            "model_name": args.model_name,
            "evaluation_metric": args.evaluation_metric if len(args.evaluation_metric) > 1 else args.evaluation_metric[0],
            "max_turns": args.max_turns,
            "max_retries": 2
        }
    )

    # Run with the same parallel rollout logic, fallback to sequential if Manager is blocked
    try:
        run_parallel_rollout(config, benchmark)
        print("Science parallel rollout completed successfully")
    except Exception as e:
        print(f"[WARN] Parallel rollout unavailable ({e}). Falling back to sequential.")
        run_science_rollout_sequential(config, benchmark)
        print("Science sequential rollout completed successfully")
