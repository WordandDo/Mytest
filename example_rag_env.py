#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
示例脚本：使用 HttpMCPRagEnv 运行 RAG 任务

这个脚本展示了如何配置和运行 RAG-only 环境来执行基准测试任务。
"""

import os
import sys
import logging

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from benchmark.benchmark import Benchmark
from run_parallel_rollout import ParallelRolloutConfig, run_parallel_rollout

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """运行 RAG 环境示例"""

    # ========================================
    # 1. 配置参数
    # ========================================

    # 数据文件路径
    data_path = "src/data/rag_demo.jsonl"

    # 并行度（Worker 数量）
    num_rollouts = 2

    # 环境模式 - 使用 RAG 专用环境
    env_mode = "http_mcp_rag"

    # 输出目录
    output_dir = "results/rag_example"

    # MCP Server 配置
    mcp_server_url = os.getenv("MCP_SERVER_URL", "http://localhost:8080")

    # Gateway 配置文件
    gateway_config_path = "gateway_config.json"

    # ========================================
    # 2. Agent 配置
    # ========================================

    agent_config = {
        "model_name": "gpt-4.1-2025-04-14",      # LLM 模型
        "max_turns": 15,                          # 最大对话轮次
        "max_retries": 2,                         # 失败重试次数
        "evaluation_metric": "exact_match",       # 评估指标
        "task_timeout": 600,                      # 任务超时（秒）
    }

    # ========================================
    # 3. 环境特定配置
    # ========================================

    env_kwargs = {
        "mcp_server_url": mcp_server_url,
        "gateway_config_path": gateway_config_path,

        # OpenAI API 配置（也可以通过环境变量设置）
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "openai_api_url": os.getenv("OPENAI_API_URL"),
        "openai_timeout": float(os.getenv("OPENAI_TIMEOUT", "30")),
        "openai_max_retries": int(os.getenv("OPENAI_MAX_RETRIES", "2")),

        # 观察类型（RAG 环境通常不需要视觉观察）
        "observation_type": "text",
    }

    # ========================================
    # 4. 加载基准测试数据
    # ========================================

    logger.info(f"Loading benchmark data from: {data_path}")
    benchmark = Benchmark(
        data_path=data_path,
        name="RAG Demo Benchmark",
        description="Example benchmark for RAG environment"
    )

    logger.info(f"Loaded {len(benchmark.get_items())} benchmark items")

    # 显示前几个问题作为示例
    logger.info("Sample questions:")
    for i, item in enumerate(benchmark.get_items()[:3], 1):
        logger.info(f"  {i}. {item.question}")
        logger.info(f"     Answer: {item.answer}")

    # ========================================
    # 5. 创建并行执行配置
    # ========================================

    config = ParallelRolloutConfig(
        num_rollouts=num_rollouts,
        env_mode=env_mode,
        output_dir=output_dir,
        env_kwargs=env_kwargs,
        agent_config_dict=agent_config
    )

    # ========================================
    # 6. 运行并行执行
    # ========================================

    logger.info("=" * 60)
    logger.info("Starting RAG Environment Execution")
    logger.info(f"Environment: {env_mode}")
    logger.info(f"Workers: {num_rollouts}")
    logger.info(f"Tasks: {len(benchmark.get_items())}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 60)

    try:
        results = run_parallel_rollout(config, benchmark)

        # ========================================
        # 7. 显示结果摘要
        # ========================================

        logger.info("=" * 60)
        logger.info("Execution Completed Successfully!")
        logger.info("=" * 60)

        # 统计成功率
        worker_results = results["worker_results"]
        total_tasks = len(worker_results)
        successful_tasks = sum(1 for r in worker_results if r.get("success", False))

        logger.info(f"Total Tasks: {total_tasks}")
        logger.info(f"Successful: {successful_tasks}")
        logger.info(f"Failed: {total_tasks - successful_tasks}")
        logger.info(f"Success Rate: {successful_tasks/total_tasks*100:.1f}%")

        # 评估结果
        eval_results = results["benchmark_evaluation"]
        if eval_results:
            avg_score = sum(r.score for r in eval_results) / len(eval_results)
            perfect_matches = sum(1 for r in eval_results if r.score == 1.0)

            logger.info(f"\nEvaluation Metrics:")
            logger.info(f"  Metric: {agent_config['evaluation_metric']}")
            logger.info(f"  Average Score: {avg_score:.4f}")
            logger.info(f"  Perfect Matches: {perfect_matches}/{len(eval_results)}")

        logger.info(f"\nResults saved to: {output_dir}")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Execution failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
