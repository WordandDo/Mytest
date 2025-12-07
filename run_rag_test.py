#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 环境测评启动脚本
使用 http_mcp_rag 环境，支持 exact_match 和 f1_score 测评
"""
import os
import sys
import logging
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from benchmark import Benchmark
from run_parallel_rollout import ParallelRolloutConfig, run_parallel_rollout

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """主函数：运行 RAG 环境测评"""

    # ==========================================
    # 1. 配置参数
    # ==========================================

    # 数据集路径
    DATA_PATH = os.getenv("DATA_PATH", "src/data/rag_demo.jsonl")

    # 并行度（worker 数量）
    NUM_ROLLOUTS = int(os.getenv("NUM_ROLLOUTS", "5"))

    # 输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", f"results/rag_test_{timestamp}")

    # 模型配置
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-2025-04-14")
    MAX_TURNS = int(os.getenv("MAX_TURNS", "15"))

    # MCP 服务器配置
    MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8080")
    RESOURCE_API_URL = os.getenv("RESOURCE_API_URL", "http://localhost:8000")

    # Gateway 配置文件
    GATEWAY_CONFIG_PATH = os.getenv("GATEWAY_CONFIG_PATH", "gateway_config.json")

    # 测评指标
    EVALUATION_METRICS = ["exact_match", "f1_score"]

    # ==========================================
    # 2. 打印配置信息
    # ==========================================

    logger.info("=" * 60)
    logger.info("RAG Benchmark Configuration")
    logger.info("=" * 60)
    logger.info(f"Data Path: {DATA_PATH}")
    logger.info(f"Num Rollouts: {NUM_ROLLOUTS}")
    logger.info(f"Output Dir: {OUTPUT_DIR}")
    logger.info(f"Model Name: {MODEL_NAME}")
    logger.info(f"Max Turns: {MAX_TURNS}")
    logger.info(f"MCP Server: {MCP_SERVER_URL}")
    logger.info(f"Resource API: {RESOURCE_API_URL}")
    logger.info(f"Gateway Config: {GATEWAY_CONFIG_PATH}")
    logger.info(f"Evaluation Metrics: {', '.join(EVALUATION_METRICS)}")
    logger.info("=" * 60)
    logger.info("")

    # ==========================================
    # 3. 加载 Benchmark 数据
    # ==========================================

    if not os.path.exists(DATA_PATH):
        logger.error(f"Data file not found: {DATA_PATH}")
        return 1

    logger.info(f"Loading benchmark data from {DATA_PATH}...")
    benchmark = Benchmark(
        data_path=DATA_PATH,
        name="RAG Benchmark",
        description="RAG environment testing with exact_match and f1_score metrics"
    )
    logger.info(f"Loaded {len(benchmark.items)} test items")
    logger.info("")

    # ==========================================
    # 4. 配置环境参数
    # ==========================================

    env_kwargs = {
        "model_name": MODEL_NAME,
        "mcp_server_url": MCP_SERVER_URL,
        "resource_api_url": RESOURCE_API_URL,
        "gateway_config_path": GATEWAY_CONFIG_PATH,
    }

    # Agent 配置
    agent_config_dict = {
        "model_name": MODEL_NAME,
        "max_turns": MAX_TURNS,
        "max_retries": 2,
        "evaluation_metric": EVALUATION_METRICS,  # 传递多个测评指标
    }

    # Rollout 配置
    config = ParallelRolloutConfig(
        num_rollouts=NUM_ROLLOUTS,
        env_mode="http_mcp_rag",  # 使用 RAG 环境
        output_dir=OUTPUT_DIR,
        env_kwargs=env_kwargs,
        agent_config_dict=agent_config_dict
    )

    # ==========================================
    # 5. 运行测评
    # ==========================================

    try:
        logger.info("Starting parallel rollout...")
        results = run_parallel_rollout(config, benchmark)

        logger.info("")
        logger.info("=" * 60)
        logger.info("✅ Benchmark completed successfully!")
        logger.info("=" * 60)
        logger.info(f"Total tasks: {len(results['worker_results'])}")
        logger.info(f"Output directory: {OUTPUT_DIR}")
        logger.info("")

        # 打印各指标的结果
        logger.info("Evaluation Results:")
        for metric, stats in results['metrics_statistics'].items():
            logger.info(f"  [{metric}]")
            logger.info(f"    Average Score: {stats['average_score']:.4f}")
            logger.info(f"    Success Rate: {stats['success_rate']:.2%}")
            logger.info(f"    Successful: {stats['successful_items']}/{stats['total_items']}")

        logger.info("=" * 60)
        return 0

    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
