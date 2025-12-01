# src/run_parallel_rollout.py
# -*- coding: utf-8 -*-
"""
并行 Rollout 框架 - 支持重资产管理的并行任务执行

[修改说明]
为了适配 HTTP MCP 多租户模式，在 run_rollout_worker 中增加了 worker_id 的自动注入逻辑。
"""
import time
import os
import sys
import json
import logging
import signal
from datetime import datetime
from multiprocessing import Manager, Process
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from benchmark import Benchmark
from envs.enviroment import Environment
from envs.factory import get_environment_class

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 预加载环境变量
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")
if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)
    logger.info(f"Loaded environment variables from {ENV_PATH}")
else:
    logger.warning(f".env file not found at {ENV_PATH}, skipping environment variable preload")

_MAIN_RESOURCE_MANAGER: Optional[Any] = None


def _register_main_signal_handlers(global_resources: Any):
    """
    注册主进程信号处理，确保 Ctrl+C 时优雅关闭资源
    """
    global _MAIN_RESOURCE_MANAGER
    _MAIN_RESOURCE_MANAGER = global_resources

    def handle_signal(signum, frame):
        logger.info(f"Main process received signal {signum}, cleaning up resources...")
        try:
            if _MAIN_RESOURCE_MANAGER and hasattr(_MAIN_RESOURCE_MANAGER, 'stop_all'):
                _MAIN_RESOURCE_MANAGER.stop_all()
        except Exception as exc:
            logger.error(f"Failed to stop resources during signal handling: {exc}")
        finally:
            sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)


@dataclass
class ParallelRolloutConfig:
    """并行 Rollout 配置"""
    num_rollouts: int = 5          # 并行度（Worker 数量）
    env_mode: str = "osworld"      # 环境模式
    env_kwargs: Dict[str, Any] = field(default_factory=dict)  # 环境配置参数
    agent_config_dict: Dict[str, Any] = field(default_factory=dict)  # Agent 配置
    output_dir: str = "results"    # 输出目录


def run_parallel_rollout(
    config: ParallelRolloutConfig,
    benchmark: Benchmark
):
    """
    运行并行 Rollout 框架
    """
    logger.info("=" * 60)
    logger.info("Starting Parallel Rollout Framework")
    logger.info(f"  Num Rollouts: {config.num_rollouts}")
    logger.info(f"  Env Mode: {config.env_mode}")
    logger.info(f"  Benchmark Items: {len(benchmark.get_items())}")
    logger.info("=" * 60)
    
    # 1. 根据 env_mode 动态获取环境类
    EnvClass = get_environment_class(config.env_mode)
    logger.info(f"Using environment class: {EnvClass.__name__}")
    
    # 2. 调用环境类的 setup_global_resources 方法初始化全局资源
    # 注意：在 HttpMCPEnv 模式下，这里通常返回 NoResourceManager，
    # 因为资源管理已移交给了独立的 Resource API 服务。
    global_resources = EnvClass.setup_global_resources(config)
    _register_main_signal_handlers(global_resources)
    
    try:
        # 3. 创建跨进程共享的数据结构
        with Manager() as manager:
            task_queue = manager.Queue()
            shared_results = manager.list()
            worker_instance_map = manager.dict()
            worker_instance_events = manager.list()
            
            # 将所有基准测试项放入任务队列
            for item in benchmark.get_items():
                task_dict = {
                    "id": item.id,
                    "question": item.question,
                    "answer": item.answer,
                    "metadata": item.metadata or {}
                }
                task_queue.put(task_dict)

            # 添加哨兵值 (Poison Pill)
            for _ in range(config.num_rollouts):
                task_queue.put(None)
            
            # 4. 启动 Worker 进程
            env_class_name = EnvClass.__module__ + "." + EnvClass.__name__
            processes = []
            for i in range(config.num_rollouts):
                # 生成唯一的 worker_id
                worker_id = f"worker-{i+1}"
                
                proc = Process(
                    target=run_rollout_worker,
                    args=(
                        worker_id,
                        task_queue,
                        env_class_name,
                        config.env_kwargs,
                        config.agent_config_dict,
                        config.output_dir,
                        global_resources,
                        config.num_rollouts,
                        shared_results,
                        worker_instance_map,
                        worker_instance_events,
                    )
                )
                proc.start()
                processes.append(proc)
                logger.info(f"Started worker process: {worker_id}")
            
            # 等待所有 Worker 进程执行完毕
            for proc in processes:
                proc.join()
            
            # 5. 收集结果并评测
            results = list(shared_results)
            worker_instance_snapshot = {k: dict(v) if isinstance(v, dict) else v for k, v in worker_instance_map.items()}
            worker_instance_events_log = list(worker_instance_events)
            
            logger.info(f"All workers completed. Total results: {len(results)}")
            
            predictions = {
                result["task_id"]: result.get("answer", "")
                for result in results
                if result.get("success", False)
            }
            
            evaluation_metric = config.agent_config_dict.get("evaluation_metric", "exact_match")
            benchmark_results = benchmark.evaluate(
                predictions=predictions,
                metric=evaluation_metric
            )
            
            logger.info("=" * 60)
            logger.info("Benchmark Evaluation Results")
            logger.info(f"  Metric: {evaluation_metric}")
            logger.info(f"  Total Items: {len(benchmark_results)}")
            if benchmark_results:
                avg_score = sum(r.score for r in benchmark_results) / len(benchmark_results)
                logger.info(f"  Average Score: {avg_score:.4f}")
            logger.info("=" * 60)
            
            # 保存结果
            os.makedirs(config.output_dir, exist_ok=True)
            
            mapping_file = os.path.join(config.output_dir, "worker_instance_map.json")
            with open(mapping_file, "w", encoding="utf-8") as f:
                json.dump(worker_instance_snapshot, f, indent=2, ensure_ascii=False)

            events_file = os.path.join(config.output_dir, "worker_instance_events.jsonl")
            with open(events_file, "w", encoding="utf-8") as f:
                for event in worker_instance_events_log:
                    json.dump(event, f, ensure_ascii=False)
                    f.write("\n")

            # =================================================================
            # [新增] 保存完整轨迹 (Trajectory)
            # 这包含了 task 信息、Agent 与 MCP 工具交互的全过程 (VM/RAG 参数及输出)
            # =================================================================
            trajectory_file = os.path.join(config.output_dir, "trajectory.jsonl")
            logger.info(f"Saving execution trajectories to {trajectory_file}...")
            with open(trajectory_file, "w", encoding="utf-8") as f:
                for res in results:
                    # res 是 environment.run_task 的返回结果，通常包含 'messages' 字段
                    json.dump(res, f, ensure_ascii=False)
                    f.write("\n")
            # =================================================================

            return {
                "worker_results": results,
                "benchmark_evaluation": benchmark_results
            }
    finally:
        try:
            if global_resources is not None and hasattr(global_resources, 'stop_all'):
                logger.info("Stopping all resources...")
                global_resources.stop_all()
        except Exception as exc:
            logger.error(f"Failed to stop resources: {exc}")


def get_active_resource_configs(environment, task_item):
    """
    根据环境实际启用的资源类型筛选任务中的资源配置
    
    Args:
        environment: Environment 实例
        task_item: BenchmarkItem 任务项
        
    Returns:
        dict: 过滤后的资源配置字典
    """
    # 1. 获取 Task 中定义的所有资源配置
    # benchmark.py 将其存放在 metadata 中
    raw_configs = task_item.get("metadata", {}).get("resource_configs", {})
    
    # 2. 获取当前环境实际启用的资源类型
    # HttpMCPEnv 从 gateway_config.json 加载了 active_resources (如 ['vm', 'rag'])
    active_types = getattr(environment, "active_resources", [])
    
    # 3. 仅保留已启用资源的配置
    active_configs = {}
    for res_type in active_types:
        if res_type in raw_configs:
            active_configs[res_type] = raw_configs[res_type]
            
    return active_configs


def run_rollout_worker(
    worker_id: str,
    task_queue,
    env_class_name: str,
    env_kwargs: Dict[str, Any],
    agent_config_dict: Dict[str, Any],
    output_dir: str,
    global_resources: Any,
    parallel_degree: int,
    shared_results,
    worker_instance_map=None,
    worker_instance_events=None,
):
    """
    Rollout Worker 进程函数
    """
    # 配置该进程的日志
    logger = logging.getLogger(f"worker.{worker_id}")
    environment: Optional[Environment] = None

    def worker_signal_handler(signum, frame):
        logger.info(f"Worker {worker_id} received signal {signum}. Cleaning up...")
        try:
            if environment:
                cleanup_fn = getattr(environment, "cleanup", None)
                if callable(cleanup_fn):
                    cleanup_fn(worker_id)
                else:
                    environment.env_close()
        except Exception as e:
            logger.error(f"Error during signal cleanup: {e}")
        sys.exit(0)

    signal.signal(signal.SIGINT, worker_signal_handler)
    signal.signal(signal.SIGTERM, worker_signal_handler)

    try:
        # 1. 动态导入环境类
        module_name, class_name = env_class_name.rsplit(".", 1)
        env_module = __import__(module_name, fromlist=[class_name])
        EnvClass = getattr(env_module, class_name)
        
        # 2. [关键修改] 将 worker_id 注入到环境参数中
        # 这是为了让 HttpMCPEnv 能够知道当前是哪个 Worker，从而在 HTTP 请求中带上 worker_id
        local_env_kwargs = env_kwargs.copy()
        local_env_kwargs["worker_id"] = worker_id

        # 3. 初始化环境实例
        environment = EnvClass(
            resource_manager=global_resources,
            parallel_degree=parallel_degree,
            **local_env_kwargs,  # 使用注入了 worker_id 的参数
        )
        
        if environment is None:
            raise RuntimeError(f"Failed to create environment instance: {env_class_name}")

        # 4. 启动环境
        try:
            environment.env_start()
        except Exception as exc:
            logger.warning(f"Worker {worker_id} env_start() failed: {exc}")

        # 调用可选的 init 方法
        init_fn = getattr(environment, "init", None)
        if callable(init_fn):
            try:
                init_fn()
            except Exception as exc:
                logger.warning(f"Worker {worker_id} environment init() failed: {exc}")

        logger.info(f"Worker {worker_id} started")

        # 5. 检查环境功能特性
        task_config_fn = getattr(environment, "initialize_with_task_config", None)
        env_supports_task_config = callable(task_config_fn)
        allocate_fn = getattr(environment, "allocate_resource", None)
        release_fn = getattr(environment, "release_resource", None)
        env_has_heavy_resource = bool(getattr(environment, "has_heavy_resource", False) and callable(allocate_fn))

        # 6. 准备 Agent 配置
        agent_config = dict(agent_config_dict)
        agent_config["output_dir"] = output_dir

        # 7. 主任务循环
        while True:
            try:
                task = task_queue.get()
                if task is None: # 哨兵值
                    logger.info(f"Worker {worker_id} received sentinel. Stopping loop.")
                    break
            except Exception as e:
                logger.error(f"Worker {worker_id} error getting task: {e}")
                break

            task_id = task.get("id", "unknown")
            resource_allocated = False
            current_resource_id = None
            task_start_time = time.time()
            logger.info(f"▶️ Worker {worker_id} START Task {task_id}")
            try:
                # 补充 metadata
                if "metadata" not in task or not isinstance(task.get("metadata"), dict):
                    task["metadata"] = task.get("metadata") or {}
                
                metadata = task.get("metadata", {})
                for key in ("config", "evaluator"):
                    if key not in task and key in metadata:
                        task[key] = metadata[key]

                # 环境特定配置
                if env_supports_task_config:
                    task_env_config = (
                        task.get("env_config")
                        or task.get("metadata", {}).get("env_config")
                    )
                    if task_env_config:
                        task_config_fn(task_env_config)

                # 重型资源分配（HttpMCPEnv 模式下通常不需要这个，因为资源分配内聚在 env_start/run_task 中了）
                # 但保留此逻辑以兼容原有的 OSWorld 环境
                if env_has_heavy_resource:
                    logger.info(f"[worker {worker_id}] requesting resource from manager")
                    # 获取活动资源配置
                    active_resource_configs = get_active_resource_configs(environment, task)
                    if not allocate_fn(worker_id, active_resource_configs):
                        raise RuntimeError("Failed to allocate resource from pool")
                    resource_allocated = True
                    
                    get_allocated_fn = getattr(environment, "get_allocated_resource_id", None)
                    current_resource_id = get_allocated_fn() if callable(get_allocated_fn) else None
                    
                    if current_resource_id:
                        # [修改] 增加日志
                        logger.info(f"[worker {worker_id}] acquired resource={current_resource_id} for task {task_id}")
                        logger.info(f"[worker {worker_id}] acquired resource={current_resource_id}")
                        if worker_instance_map is not None:
                            worker_instance_map[worker_id] = {
                                "instance_id": current_resource_id,
                                "task_id": task_id,
                                "assigned_time": datetime.now().isoformat(),
                            }
                        if worker_instance_events is not None:
                            worker_instance_events.append({
                                "timestamp": datetime.now().isoformat(),
                                "worker_id": worker_id,
                                "instance_id": current_resource_id,
                                "task_id": task_id,
                                "action": "allocate",
                            })

                # 8. 执行任务
                result = environment.run_task(task, agent_config, logger)

                if shared_results is not None:
                    shared_results.append(result)
                # [修改] 记录任务耗时
                duration = time.time() - task_start_time
                status_icon = "✅" if result.get("success") else "❌"
                logger.info(f"{status_icon} Worker {worker_id} FINISH Task {task_id} in {duration:.1f}s (Success: {result.get('success')})")
                logger.info(f"Worker {worker_id} completed task {task_id}")

            except Exception as e:
                logger.error(f"Task {task_id} failed: {e}", exc_info=True)
                failure_result = {
                    "task_id": task_id,
                    "question": task.get("question", ""),
                    "answer": "",
                    "messages": [],
                    "success": False,
                    "error": str(e),
                }
                if shared_results is not None:
                    shared_results.append(failure_result)
            finally:
                # 释放重型资源（如果已分配）
                if env_has_heavy_resource and resource_allocated and callable(release_fn):
                    if current_resource_id:
                        logger.info(f"[worker {worker_id}] releasing resource={current_resource_id}")
                    release_fn(worker_id, reset=True)
                    
                    if worker_instance_map is not None:
                        worker_instance_map.pop(worker_id, None)
                    if worker_instance_events is not None:
                        worker_instance_events.append({
                            "timestamp": datetime.now().isoformat(),
                            "worker_id": worker_id,
                            "instance_id": current_resource_id,
                            "task_id": task_id,
                            "action": "release",
                        })

    finally:
        # Worker 退出清理
        if worker_instance_map is not None:
            worker_instance_map.pop(worker_id, None)
        logger.info(f"Worker {worker_id} stopped")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run parallel rollout")
    parser.add_argument("--data_path", type=str, required=True, help="Path to benchmark data file")
    parser.add_argument("--num_rollouts", type=int, default=5, help="Number of parallel workers")
    parser.add_argument("--env_mode", type=str, default="http_mcp", help="Environment mode") # 默认改为 http_mcp
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    
    # 额外参数：MCP 和 Resource API 地址
    parser.add_argument("--mcp_server_url", type=str, default="http://localhost:8080", help="MCP Server URL")
    parser.add_argument("--resource_api_url", type=str, default="http://localhost:8000", help="Resource API URL")
    
    args = parser.parse_args()
    
    # 加载 Benchmark
    benchmark = Benchmark(data_path=args.data_path)
    
    # 环境配置
    env_kwargs: Dict[str, Any] = {
        "observation_type": "screenshot_a11y_tree",
        "mcp_server_url": args.mcp_server_url,
        "resource_api_url": args.resource_api_url,
    }
    
    # 创建配置
    config = ParallelRolloutConfig(
        num_rollouts=args.num_rollouts,
        env_mode=args.env_mode,
        output_dir=args.output_dir,
        env_kwargs=env_kwargs,
        agent_config_dict={
            "model_name": "gpt-4.1-2025-04-14",
            "evaluation_metric": "exact_match"
        }
    )
    
    # 运行并行 Rollout
    results = run_parallel_rollout(config, benchmark)
    
    logger.info("Parallel rollout completed successfully")