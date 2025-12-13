"""
数据合成主Pipeline (Multi-Process / MCP Compatible)

整合trajectory采样、选择和QA合成的完整流程
已适配 HttpMCPEnv, HttpMCPRagEnv, HttpMCPSearchEnv，并修复了缺失模块的导入问题。
"""

import json
import os
import hashlib
import sys
import time
from typing import List, Dict, Callable, Optional, Set, Any, Union
from multiprocessing import Process, Manager

# ================= 🔧 新增代码开始 =================
from dotenv import load_dotenv

# 加载 .env 文件到环境变量
# verbose=True 会在找不到文件时打印警告
# override=True 确保 .env 中的值覆盖系统默认值（可选）
load_dotenv(verbose=True, override=True)
# ================= 🔧 新增代码结束 =================

# 添加源码路径到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 引入基础模型和配置
from models import TrajectoryNode, Trajectory, SynthesizedQA, SynthesizedTask
from synthesis_config import SynthesisConfig
from trajectory_sampler import GenericTrajectorySampler
from trajectory_selector import GenericTrajectorySelector
from qa_synthesizer import GenericQASynthesizer
from task_synthesizer import OSWorldTaskSynthesizer

# 引入 MCP 环境类 (直接从文件导入，避开 envs/__init__.py 中可能存在的错误引用)
try:
    from envs.http_mcp_env import HttpMCPEnv
    from envs.http_mcp_rag_env import HttpMCPRagEnv
    from envs.http_mcp_search_env import HttpMCPSearchEnv
except ImportError as e:
    print(f"❌ Critical Error: MCP Environment files missing: {e}")
    sys.exit(1)


def _generate_source_id(seed_data: str, seed_idx: int) -> str:
    """生成source的唯一标识"""
    content_hash = hashlib.md5(seed_data.encode('utf-8')).hexdigest()[:8]
    return f"src_{seed_idx:04d}_{content_hash}"


def _create_environment(config: SynthesisConfig, worker_id: Optional[str] = None) -> Union[HttpMCPEnv, HttpMCPRagEnv, HttpMCPSearchEnv]:
    """
    根据配置创建相应的环境。
    
    策略：
    由于本地缺失 Math/Python/Web 等原生环境代码，我们将这些模式
    统一映射到通用的 HttpMCPEnv，依靠 MCP Server 端加载对应工具来提供能力。
    
    Args:
        config: 合成配置
        worker_id: 进程唯一ID (用于 MCP 资源分配和日志)
    """
    mode = config.environment_mode.lower()
    kwargs = config.environment_kwargs.copy()
    kwargs['model_name'] = config.model_name
    
    # 将 worker_id 注入到 kwargs 中，供 MCP 环境使用
    if worker_id:
        kwargs['worker_id'] = worker_id
    
    # 1. RAG 专用环境
    if mode == "rag":
        return HttpMCPRagEnv(**kwargs)
        
    # 2. Search 专用环境
    elif mode == "search":
        return HttpMCPSearchEnv(**kwargs)
        
    # 3. 通用 MCP 环境 (处理 Math, Python, Web, OSWorld 等)
    elif mode in ["mcp", "http_mcp", "math", "python", "py", "web", "osworld", "gui"]:
        return HttpMCPEnv(**kwargs)
        
    else:
        raise ValueError(f"不支持的环境模式: {mode} (且未找到对应的 MCP 映射)")


def run_synthesis_worker(
    worker_id: str,
    task_queue: Any,  # Manager.Queue() proxy object
    config: SynthesisConfig,
    file_lock: Any,  # Manager.Lock() proxy object
    qa_saver: Callable[[List[Dict]], None],
    traj_saver: Callable[[List[Dict]], None]
):
    """
    Worker 进程函数：并行处理 Seeds，包含 MCP 资源生命周期管理。
    """
    print(f"\n[Worker {worker_id}] Starting up...")

    # 1. 初始化环境
    try:
        environment = _create_environment(config, worker_id=worker_id)
    except Exception as e:
        print(f"[Worker {worker_id}] ❌ Failed to create environment: {e}")
        return

    # 2. 【关键修改】先启动环境连接，确保能获取工具列表
    if hasattr(environment, "env_start") and callable(environment.env_start):
        try:
            environment.env_start()
            # print(f"[Worker {worker_id}] Connected to Gateway") # 可选日志
        except Exception as e:
            print(f"[Worker {worker_id}] env_start() failed: {e}")

    # 3. 【关键修改】环境连接后再初始化 Sampler
    sampler = GenericTrajectorySampler(
        environment=environment,
        config=config
    )
    
    selector = GenericTrajectorySelector(config=config)
    
    # 初始化合成器
    if config.output_format == "task":
        synthesizer = OSWorldTaskSynthesizer(config=config)
    else:
        synthesizer = GenericQASynthesizer(config=config)

    # 检查是否需要每任务资源分配 (Heavy Resource Check)
    # HttpMCPEnv 默认为 True, HttpMCPSearchEnv active_resources 为空，allocate 会快速返回 True
    env_has_heavy_resource = bool(getattr(environment, "has_heavy_resource", False) and callable(getattr(environment, "allocate_resource", None)))

    # 2. 主任务循环 (Pull Model)
    while True:
        try:
            # 尝试从队列获取任务
            seed_task = task_queue.get(timeout=30) 
            
            if seed_task is None: # 哨兵值
                print(f"[Worker {worker_id}] Received sentinel. Stopping loop.")
                break
        except Exception as e:
            # 队列超时或空
            break

        seed_data = seed_task["seed_data"]
        seed_idx = seed_task["seed_idx"]
        source_id = _generate_source_id(seed_data, seed_idx)
        
        print(f"\n{'#'*60}")
        print(f"[Worker {worker_id}] START Seed {seed_idx}, Source ID: {source_id}")
        print(f"{'#'*60}\n")
        
        resource_allocated = False
        
        try:
            # --- 显式资源分配 (MCP Resource Lifecycle) ---
            if env_has_heavy_resource:
                # print(f"[Worker {worker_id}] 🔐 Requesting resource via MCP...")
                if not environment.allocate_resource(worker_id):
                     raise RuntimeError(f"Failed to allocate resource via MCP for {worker_id}")
                resource_allocated = True
                # print(f"[Worker {worker_id}] ✅ Resource ready.")
            
            # Step 1: Trajectory Sampling
            # print(f"\n📊 步骤 1/3: Trajectory Sampling")
            trajectory_tree = sampler.sample_trajectory_tree(seed_data)
            
            # Step 2: Trajectory Selection
            # print(f"\n🎯 步骤 2/3: Trajectory Selection")
            if not sampler.root_id:
                raise RuntimeError(f"[Worker {worker_id}] Sampler root_id is None after sampling")

            selected_trajectories = selector.select_trajectories(
                nodes=trajectory_tree,
                root_id=sampler.root_id,
                seed_data=seed_data,
                source_id=source_id,
                max_selected_traj=config.max_selected_traj
            )
            
            # Step 3: 数据合成（QA或Task）
            outputs = []
            output_type = "QA对" if config.output_format != "task" else "任务"
            
            # print(f"\n✨ 步骤 3/3: {output_type} Synthesis")
            for qa_idx, trajectory in enumerate(selected_trajectories):
                try:
                    if config.output_format == "task":
                        # OSWorldTaskSynthesizer has synthesize_task method
                        if hasattr(synthesizer, 'synthesize_task'):
                            synthesized_output = synthesizer.synthesize_task(trajectory, qa_idx)  # type: ignore
                        else:
                            raise AttributeError(f"Synthesizer does not have 'synthesize_task' method")
                    else:
                        # GenericQASynthesizer has synthesize_qa method
                        if hasattr(synthesizer, 'synthesize_qa'):
                            synthesized_output = synthesizer.synthesize_qa(trajectory, qa_idx)  # type: ignore
                        else:
                            raise AttributeError(f"Synthesizer does not have 'synthesize_qa' method")

                    if synthesized_output:
                        outputs.append(synthesized_output.to_dict())
                except Exception as e:
                    print(f"[Worker {worker_id}] ❌ 合成失败 (轨迹 {qa_idx}): {str(e)}")
            
            trajectories_data = [traj.to_dict() for traj in selected_trajectories]
            
            print(f"[Worker {worker_id}] ✅ Seed {seed_idx} 完成! 生成 {len(outputs)} {output_type}")
            
            # --- 实时保存结果 (使用锁) ---
            if outputs:
                qa_saver(outputs) 
            if trajectories_data:
                traj_saver(trajectories_data) 
                
        except Exception as e:
            error_msg = f"[Worker {worker_id}] ❌ Seed {seed_idx} 失败: {str(e)}"
            print(f"\n{error_msg}")
            # import traceback
            # traceback.print_exc()
            
        finally:
            # --- 显式资源释放 (MCP Resource Lifecycle) ---
            if env_has_heavy_resource and resource_allocated:
                # print(f"[Worker {worker_id}] ♻️ Releasing resource...")
                try:
                    environment.release_resource(worker_id, reset=True)
                except Exception as e:
                    print(f"[Worker {worker_id}] ⚠️ Error releasing resource: {e}")
            
    # Worker 退出时关闭环境连接
    if hasattr(environment, "env_close") and callable(environment.env_close):
        environment.env_close()
    print(f"[Worker {worker_id}] Stopped.")


class GenericDataSynthesis:
    """
    通用数据合成主类 - 支持所有环境和工具
    """
    
    def __init__(self, config: SynthesisConfig, output_dir: str = "synthesis_results"):
        """
        初始化通用数据合成系统
        """
        self.config = config
        self.output_dir = output_dir
        
        # 验证配置
        errors = config.validate()
        if errors:
            raise ValueError(f"配置错误: {', '.join(errors)}")
        
        # 创建主环境 (用于 Main Process 获取元数据/工具列表)
        print(f"初始化 {config.environment_mode.upper()} Environment (Main Process)...")
        # 主进程使用 "main" 作为 ID
        self.environment = _create_environment(config, worker_id="main")

        # 先连接环境，确保工具列表可用
        if hasattr(self.environment, "env_start"):
            try:
                self.environment.env_start()
                print(f"✅ Main Process 已连接到 Gateway")
            except Exception as e:
                print(f"⚠️ Main Process 连接失败（非致命）: {e}")

        # 创建组件 (注意：Sampler 需要环境已连接才能获取工具列表)
        self.sampler = GenericTrajectorySampler(
            environment=self.environment,
            config=config
        )
        
        self.selector = GenericTrajectorySelector(config=config)
        
        if config.output_format == "task":
            self.synthesizer = OSWorldTaskSynthesizer(config=config)
            print(f"使用OSWorld任务合成器（输出格式：task）")
        else:
            self.synthesizer = GenericQASynthesizer(config=config)
            print(f"使用QA合成器（输出格式：qa）")
        
        self.qa_file_path = None
        self.traj_file_path = None
        self.processed_source_ids: Set[str] = set()
        self.file_lock: Optional[Any] = None  # Manager.Lock() proxy object
    
    def _initialize_output_files(self):
        """初始化输出文件路径并创建输出目录"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        if self.config.output_format == "task":
            self.qa_file_path = os.path.join(
                self.output_dir, 
                f"synthesized_tasks_{self.config.environment_mode}.jsonl"
            )
        else:
            self.qa_file_path = os.path.join(
                self.output_dir, 
                f"synthesized_qa_{self.config.environment_mode}.jsonl"
            )
        
        self.traj_file_path = os.path.join(
            self.output_dir, 
            f"trajectories_{self.config.environment_mode}.jsonl"
        )
        
        print(f"💾 输出文件: {self.qa_file_path}")
        self._load_processed_source_ids()
    
    def _load_processed_source_ids(self):
        """从已有的输出文件中加载已处理的source_id"""
        self.processed_source_ids.clear()

        if self.qa_file_path and os.path.exists(self.qa_file_path):
            try:
                with open(self.qa_file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            qa_dict = json.loads(line)
                            if "source_id" in qa_dict:
                                self.processed_source_ids.add(qa_dict["source_id"])
                
                if self.processed_source_ids:
                    print(f"🔄 发现 {len(self.processed_source_ids)} 个已处理的source，将跳过这些seed")
            except Exception as e:
                print(f"⚠️  读取已处理记录时出错: {e}")
                self.processed_source_ids.clear()
    
    def _save_qa_immediately(self, qas_dicts: List[Dict]):
        """立即将QA对追加保存到文件（进程安全）"""
        if not self.file_lock or not self.qa_file_path:
            return
        with self.file_lock:
            with open(self.qa_file_path, "a", encoding="utf-8") as f:
                for qa_dict in qas_dicts:
                    f.write(json.dumps(qa_dict, ensure_ascii=False) + "\n")
    
    def _save_trajectories_immediately(self, trajectories_data: List[Dict]):
        """立即将trajectories追加保存到文件（进程安全）"""
        if not self.file_lock or not self.traj_file_path:
            return
        with self.file_lock:
            with open(self.traj_file_path, "a", encoding="utf-8") as f:
                for traj in trajectories_data:
                    f.write(json.dumps(traj, ensure_ascii=False) + "\n")
    
    def run(self, seeds: List[str]) -> List[Dict]:
        """
        运行完整的数据合成pipeline（使用 Process/Queue 架构）
        """
        if self.config.number_of_seed is not None:
            seeds = seeds[:self.config.number_of_seed]
        
        print(f"\n{'='*80}")
        print(f"🚀 通用Agent数据合成 Pipeline 启动")
        print(f"{'='*80}")
        print(f"环境模式: {self.config.environment_mode}")
        print(f"总Seed数量: {len(seeds)}")
        print(f"并行度: {self.config.max_workers} workers")
        
        # 显示可用工具列表
        try:
            tool_names = [t['name'] for t in self.sampler.available_tools]
            print(f"可用工具 ({len(tool_names)}): {tool_names[:5] if len(tool_names) > 5 else tool_names}...")
        except Exception as e:
            print(f"Warning: Failed to list tools (Non-fatal): {e}")

        # 关闭主进程的连接（Worker 会建立自己的连接）
        try:
            if hasattr(self.environment, "env_close"):
                self.environment.env_close()
                print(f"✅ Main Process 已断开连接（Worker 将建立独立连接）")
        except Exception as e:
            print(f"⚠️ Main Process 断开连接失败（非致命）: {e}")

        print(f"{'='*80}\n")
        
        self._initialize_output_files()
        
        skipped_count = 0
        
        with Manager() as manager:
            task_queue = manager.Queue()
            self.file_lock = manager.Lock() 

            # 1. 填充任务队列
            seeds_to_process = []
            for seed_idx, seed_data in enumerate(seeds, 1):
                source_id = _generate_source_id(seed_data, seed_idx)
                
                if source_id in self.processed_source_ids:
                    skipped_count += 1
                else:
                    seeds_to_process.append({
                        "seed_idx": seed_idx,
                        "seed_data": seed_data,
                        "source_id": source_id,
                    })

            if not seeds_to_process:
                print("\n所有seed都已处理，无需继续")
                return []
            
            total_tasks = len(seeds_to_process)
            
            for task in seeds_to_process:
                task_queue.put(task)

            # 2. 添加哨兵值 (Poison Pills)
            for _ in range(self.config.max_workers):
                task_queue.put(None)

            # 3. 启动 Worker 进程
            processes = []
            for i in range(self.config.max_workers):
                worker_id = f"worker-{i+1}"
                
                proc = Process(
                    target=run_synthesis_worker,
                    args=(
                        worker_id,
                        task_queue,
                        self.config,
                        self.file_lock,
                        self._save_qa_immediately, 
                        self._save_trajectories_immediately, 
                    )
                )
                proc.start()
                processes.append(proc)
                print(f"Started worker process: {worker_id}")
            
            # 4. 等待 Worker 进程完成
            try:
                for proc in processes:
                    proc.join()
            except KeyboardInterrupt:
                print("Main process interrupted. Terminating workers...")
                for proc in processes:
                    if proc.is_alive():
                        proc.terminate()
        
        print(f"\n\n{'='*80}")
        print(f"🎉 数据合成完成!")
        print(f"{'='*80}")
        print(f"总Seed数量: {len(seeds)} 个")
        print(f"已跳过: {skipped_count} 个")
        print(f"新处理: {total_tasks} 个")
        print(f"{'='*80}\n")
        
        return []
    
    def save_results(self):
        """显示结果保存位置"""
        if not self.qa_file_path:
            return
        
        print(f"💾 QA对已保存到: {self.qa_file_path}")
        print(f"💾 Trajectories已保存到: {self.traj_file_path}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="通用Agent数据合成系统 (并行版)")
    
    parser.add_argument("--config", type=str, required=True,
                       help="配置文件路径 (.json 或 .yaml)")
    parser.add_argument("--seeds", type=str, required=True,
                       help="Seed数据JSON文件路径")
    parser.add_argument("--output-dir", type=str, default="synthesis_results",
                       help="输出目录")
    
    args = parser.parse_args()
    
    print(f"加载配置文件: {args.config}")
    if args.config.endswith('.json'):
        config = SynthesisConfig.from_json(args.config)
    elif args.config.endswith('.yaml') or args.config.endswith('.yml'):
        config = SynthesisConfig.from_yaml(args.config)
    else:
        raise ValueError("配置文件必须是 .json 或 .yaml 格式")
    
    print(f"读取 seed 数据文件: {args.seeds}")
    with open(args.seeds, "r", encoding="utf-8") as f:
        seeds = json.load(f)
    
    # 兼容单个字符串输入
    if isinstance(seeds, str):
        seeds = [seeds]
    
    # 确保是列表
    if not isinstance(seeds, list):
        raise ValueError("Seed文件格式错误")

    print(f"加载了 {len(seeds)} 个 seed 数据")
    
    synthesizer = GenericDataSynthesis(config=config, output_dir=args.output_dir)
    synthesizer.run(seeds)
    synthesizer.save_results()
    
    print(f"\n✅ 全部完成!")


if __name__ == "__main__":
    main()
