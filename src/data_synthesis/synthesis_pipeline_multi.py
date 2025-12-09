"""
数据合成主Pipeline

整合trajectory采样、选择和QA合成的完整流程
"""

import json
import os
import bdb
import hashlib
from typing import List, Dict, Tuple, Set, Callable, Optional
from multiprocessing import Process, Manager, Queue, Lock

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs import (
    Environment,
    MathEnvironment,
    PythonEnvironment,
    RAGEnvironment,
    WebEnvironment,
    OSWorldEnvironment
)
from models import TrajectoryNode, Trajectory, SynthesizedQA, SynthesizedTask
from synthesis_config import SynthesisConfig
from trajectory_sampler import GenericTrajectorySampler
from trajectory_selector import GenericTrajectorySelector
from qa_synthesizer import GenericQASynthesizer
from task_synthesizer import OSWorldTaskSynthesizer


def run_synthesis_worker(
    worker_id: str,
    task_queue: Queue,
    config: SynthesisConfig,
    file_lock: Lock,
    qa_saver: Callable[[List[Dict]], None],
    traj_saver: Callable[[List[Dict]], None]
):
    """
    Worker function to process seeds in parallel using the Process/Queue model,
    implementing explicit resource allocation/release for heavy resources.
    """
    print(f"\n[Worker {worker_id}] Starting up...")

    # 1. 初始化环境和组件 (Worker 进程独享)
    environment = _create_environment(config)
    
    sampler = GenericTrajectorySampler(
        environment=environment,
        config=config
    )
    
    selector = GenericTrajectorySelector(config=config)
    
    # 简化合成器初始化（原逻辑）
    if config.output_format == "task":
        from task_synthesizer import OSWorldTaskSynthesizer
        synthesizer = OSWorldTaskSynthesizer(config=config)
    else:
        synthesizer = GenericQASynthesizer(config=config)

    # 尝试启动环境连接 (建立 MCP 连接等)
    if hasattr(environment, "env_start") and callable(environment.env_start):
        try:
            environment.env_start()
        except Exception as e:
            print(f"[Worker {worker_id}] env_start() failed: {e}")

    # 检查是否需要每任务资源分配 (Heavy Resource Check)
    env_has_heavy_resource = bool(getattr(environment, "has_heavy_resource", False) and callable(getattr(environment, "allocate_resource", None)))

    # 2. 主任务循环 (Pull Model)
    while True:
        try:
            # 尝试从队列获取任务，设置超时以允许进程在空闲时关闭
            seed_task = task_queue.get(timeout=30) 
            
            if seed_task is None: # 哨兵值
                print(f"[Worker {worker_id}] Received sentinel. Stopping loop.")
                break
        except Exception as e:
            # Queue is empty for a long time or other error
            print(f"[Worker {worker_id}] Error getting task: {e}")
            break

        seed_data = seed_task["seed_data"]
        seed_idx = seed_task["seed_idx"]
        source_id = _generate_source_id(seed_data, seed_idx)
        
        print(f"\n{'#'*80}")
        print(f"[Worker {worker_id}] START Seed {seed_idx}, Source ID: {source_id}")
        print(f"内容: {seed_data[:100]}{'...' if len(seed_data) > 100 else ''}")
        print(f"{'#'*80}\n")
        
        resource_allocated = False
        
        try:
            # --- 显式资源分配 (Rollout 架构的核心) ---
            if env_has_heavy_resource:
                print(f"[Worker {worker_id}] 🔐 Requesting heavy resource via MCP...")
                # 使用 worker_id 作为 client ID
                if not environment.allocate_resource(worker_id):
                     raise RuntimeError("Failed to allocate resource via MCP")
                resource_allocated = True
                print(f"[Worker {worker_id}] ✅ Resource allocated.")
            
            # Step 1: Trajectory Sampling
            print(f"\n📊 步骤 1/3: Trajectory Sampling")
            trajectory_tree = sampler.sample_trajectory_tree(seed_data)
            
            # Step 2: Trajectory Selection
            print(f"\n🎯 步骤 2/3: Trajectory Selection")
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
            
            print(f"\n✨ 步骤 3/3: {output_type} Synthesis")
            for qa_idx, trajectory in enumerate(selected_trajectories):
                try:
                    if config.output_format == "task":
                        synthesized_output = synthesizer.synthesize_task(trajectory, qa_idx)
                    else:
                        synthesized_output = synthesizer.synthesize_qa(trajectory, qa_idx)
                        
                    if synthesized_output:
                        outputs.append(synthesized_output.to_dict())
                except Exception as e:
                    print(f"[Worker {worker_id}] ❌ 合成失败 (轨迹 {qa_idx}): {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            trajectories_data = [traj.to_dict() for traj in selected_trajectories]
            
            print(f"\n✅ Seed {seed_idx} 完成! 生成了 {len(outputs)} 个{output_type}")
            
            # --- 实时保存结果 (Worker 进程调用主进程传入的 Saver) ---
            if outputs:
                qa_saver(outputs) 
            if trajectories_data:
                traj_saver(trajectories_data) 
                
        except Exception as e:
            error_msg = f"❌ Seed {seed_idx} 失败: {str(e)}"
            print(f"\n{error_msg}")
            import traceback
            traceback.print_exc()
            
        finally:
            # --- 显式资源释放 (Rollout 架构的核心) ---
            if env_has_heavy_resource and resource_allocated:
                print(f"[Worker {worker_id}] ♻️ Releasing resource via MCP (reset=True)...")
                try:
                    environment.release_resource(worker_id, reset=True)
                except Exception as e:
                    print(f"[Worker {worker_id}] ⚠️ Error releasing resource: {e}")
            
    # Worker 退出时关闭环境连接
    if hasattr(environment, "env_close") and callable(environment.env_close):
        environment.env_close()
    print(f"[Worker {worker_id}] Stopped.")


# --- Keep the helper functions _generate_source_id and _create_environment here ---
# (As they were in the original file, just before the class definition)

def _generate_source_id(seed_data: str, seed_idx: int) -> str:
    """生成source的唯一标识"""
    content_hash = hashlib.md5(seed_data.encode('utf-8')).hexdigest()[:8]
    return f"src_{seed_idx:04d}_{content_hash}"


def _create_environment(config: SynthesisConfig):
    """根据配置创建相应的环境"""
    mode = config.environment_mode.lower()
    kwargs = config.environment_kwargs.copy()
    kwargs['model_name'] = config.model_name
    
    if mode == "web":
        from envs import WebEnvironment
        return WebEnvironment(**kwargs)
    elif mode == "math":
        from envs import MathEnvironment
        return MathEnvironment(**kwargs)
    elif mode == "python" or mode == "py":
        from envs import PythonEnvironment
        return PythonEnvironment(**kwargs)
    elif mode == "rag":
        if 'rag_index' not in kwargs:
            raise ValueError("RAG环境需要提供rag_index参数")
        from envs import RAGEnvironment
        return RAGEnvironment(**kwargs)
    elif mode == "osworld" or mode == "gui":
        # OSWorld/GUI环境需要VM配置
        required_params = ['path_to_vm']
        missing = [p for p in required_params if p not in kwargs]
        if missing:
            raise ValueError(f"OSWorld环境需要提供以下参数: {', '.join(missing)}")
        from envs import OSWorldEnvironment
        return OSWorldEnvironment(**kwargs)
    else:
        raise ValueError(f"不支持的环境模式: {mode}")


class GenericDataSynthesis:
    """
    通用数据合成主类 - 支持所有环境和工具
    """
    
    def __init__(self, config: SynthesisConfig, output_dir: str = "synthesis_results"):
        """
        初始化通用数据合成系统
        
        Args:
            config: 合成配置
            output_dir: 输出目录
        """
        self.config = config
        self.output_dir = output_dir
        
        # 验证配置
        errors = config.validate()
        if errors:
            raise ValueError(f"配置错误: {', '.join(errors)}")
        
        # 创建环境
        print(f"初始化 {config.environment_mode.upper()} Environment...")
        self.environment = self._create_environment()
        
        # 创建三个组件
        self.sampler = GenericTrajectorySampler(
            environment=self.environment,
            config=config
        )
        
        self.selector = GenericTrajectorySelector(config=config)
        
        # 根据输出格式选择合成器
        if config.output_format == "task":
            self.synthesizer = OSWorldTaskSynthesizer(config=config)
            print(f"使用OSWorld任务合成器（输出格式：task）")
        else:
            self.synthesizer = GenericQASynthesizer(config=config)
            print(f"使用QA合成器（输出格式：qa）")
        
        # 存储结果 (移除内存列表，依赖文件实时写入)
        # self.trajectory_tree: Dict[str, TrajectoryNode] = {}
        # self.selected_trajectories: List[Trajectory] = []
        # self.synthesized_qas: List[SynthesizedQA] = []  # QA格式
        # self.synthesized_tasks: List[SynthesizedTask] = []  # Task格式
        
        # 初始化输出文件路径（在run时创建）
        self.qa_file_path = None
        self.traj_file_path = None
        
        # 已处理的source_id集合
        self.processed_source_ids: Set[str] = set()
        
        # 文件写入锁（在 run() 中使用 Manager.Lock() 进行初始化）
        self.file_lock: Optional[Lock] = None
    
    def _create_environment(self) -> Environment:
        """根据配置创建相应的环境"""
        return _create_environment(self.config)
    
    def _initialize_output_files(self):
        """初始化输出文件路径并创建输出目录"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 根据输出格式设置文件路径
        if self.config.output_format == "task":
            # OSWorld任务格式
            self.qa_file_path = os.path.join(
                self.output_dir, 
                f"synthesized_tasks_{self.config.environment_mode}.jsonl"
            )
        else:
            # QA对格式
            self.qa_file_path = os.path.join(
                self.output_dir, 
                f"synthesized_qa_{self.config.environment_mode}.jsonl"
            )
        
        # 设置trajectories输出文件路径（固定文件名，支持断点续传）
        self.traj_file_path = os.path.join(
            self.output_dir, 
            f"trajectories_{self.config.environment_mode}.jsonl"
        )
        
        print(f"💾 输出文件: {self.qa_file_path}")
        print(f"💾 输出格式: {self.config.output_format}")
        
        # 加载已处理的source_id
        self._load_processed_source_ids()
    
    def _load_processed_source_ids(self):
        """从已有的输出文件中加载已处理的source_id"""
        self.processed_source_ids.clear()
        
        # 从QA文件中读取已处理的source_id
        if os.path.exists(self.qa_file_path):
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
        with self.file_lock:
            with open(self.qa_file_path, "a", encoding="utf-8") as f:
                for qa_dict in qas_dicts:
                    f.write(json.dumps(qa_dict, ensure_ascii=False) + "\n")
    
    def _save_trajectories_immediately(self, trajectories_data: List[Dict]):
        """立即将trajectories追加保存到文件（进程安全）"""
        with self.file_lock:
            with open(self.traj_file_path, "a", encoding="utf-8") as f:
                for traj in trajectories_data:
                    f.write(json.dumps(traj, ensure_ascii=False) + "\n")
    
    def run(self, seeds: List[str]) -> List[Dict]:
        """
        运行完整的数据合成pipeline（使用 Process/Queue 架构）
        
        Args:
            seeds: Seed数据列表
            
        Returns:
            合成的QA对字典列表 (为了兼容性返回空列表，实际结果已写入文件)
        """
        # 根据配置限制处理的seed数量
        if self.config.number_of_seed is not None:
            seeds = seeds[:self.config.number_of_seed]
        
        print(f"\n{'='*80}")
        print(f"🚀 通用Agent数据合成 Pipeline 启动")
        print(f"{'='*80}")
        print(f"环境模式: {self.config.environment_mode}")
        print(f"Seed说明: {self.config.seed_description or '(未指定)'}")
        print(f"可用工具: {[t['name'] for t in self.sampler.available_tools]}")
        print(f"总Seed数量: {len(seeds)}")
        print(f"并行度: {self.config.max_workers} workers")
        print(f"模型: {self.config.model_name}")
        print(f"{'='*80}\n")
        
        # 初始化输出文件
        self._initialize_output_files()
        
        skipped_count = 0
        
        # --- 替换 ProcessPoolExecutor 架构 ---
        with Manager() as manager:
            task_queue = manager.Queue()
            
            # 进程安全锁 (用于文件 I/O)
            self.file_lock = manager.Lock() 

            # 1. 填充任务队列，并处理断点续传
            seeds_to_process = []
            for seed_idx, seed_data in enumerate(seeds, 1):
                source_id = _generate_source_id(seed_data, seed_idx)
                
                if source_id in self.processed_source_ids:
                    skipped_count += 1
                    # print(f"\n⏭️  跳过 Seed {seed_idx}/{len(seeds)} (已处理: {source_id})")
                else:
                    seeds_to_process.append({
                        "seed_idx": seed_idx,
                        "seed_data": seed_data,
                        "source_id": source_id,
                    })

            if not seeds_to_process:
                print("\n所有seed都已处理，无需继续")
            
            total_tasks = len(seeds_to_process)
            
            # 将任务放入队列
            for task in seeds_to_process:
                task_queue.put(task)

            # 2. 添加哨兵值 (Poison Pill)
            for _ in range(self.config.max_workers):
                task_queue.put(None)

            # 3. 启动 Worker 进程
            processes = []
            for i in range(self.config.max_workers):
                worker_id = f"worker-{i+1}"
                
                # 启动 Worker 进程并传入共享资源和方法
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
            
            # 4. 等待 Worker 进程完成 (Join)
            try:
                for proc in processes:
                    proc.join()
            except KeyboardInterrupt:
                print("Main process interrupted. Terminating workers...")
                for proc in processes:
                    if proc.is_alive():
                        proc.terminate()
            
        # 5. 清理（原代码中 cleanup 放在 finally 块中，这里保持不变）
        # self.cleanup() # This is the final step outside the Manager block

        # Final statistics based on total tasks processed (approximation)
        newly_processed_count = total_tasks
        
        print(f"\n\n{'='*80}")
        print(f"🎉 数据合成完成!")
        print(f"{'='*80}")
        print(f"总Seed数量: {len(seeds)} 个")
        print(f"已跳过: {skipped_count} 个")
        print(f"新处理: {newly_processed_count} 个")
        print(f"{'='*80}\n")
        
        # 返回空列表，兼容调用者
        return []
    
    def save_results(self):
        """显示结果保存位置（QA对和trajectories已实时保存）"""
        if not self.qa_file_path:
            print("⚠️  警告: 没有运行过pipeline，无法保存结果")
            return
        
        print(f"💾 QA对已保存到: {self.qa_file_path}")
        print(f"💾 Trajectories已保存到: {self.traj_file_path}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="通用Agent数据合成系统")
    
    parser.add_argument("--config", type=str, required=True,
                       help="配置文件路径 (.json 或 .yaml)")
    parser.add_argument("--seeds", type=str, required=True,
                       help="Seed数据JSON文件路径（支持任意类型的seed：entity/problem/text/url等）")
    parser.add_argument("--output-dir", type=str, default="synthesis_results",
                       help="输出目录")
    
    args = parser.parse_args()
    
    # 加载配置
    print(f"加载配置文件: {args.config}")
    if args.config.endswith('.json'):
        config = SynthesisConfig.from_json(args.config)
    elif args.config.endswith('.yaml') or args.config.endswith('.yml'):
        config = SynthesisConfig.from_yaml(args.config)
    else:
        raise ValueError("配置文件必须是 .json 或 .yaml 格式")
    
    # 读取seed数据（简单字符串列表）
    print(f"读取 seed 数据文件: {args.seeds}")
    with open(args.seeds, "r", encoding="utf-8") as f:
        seeds = json.load(f)
        if not isinstance(seeds, list):
            raise ValueError("Seed文件格式错误：必须是字符串列表，例如: [\"seed1\", \"seed2\", \"seed3\"]")
        if not all(isinstance(s, str) for s in seeds):
            raise ValueError("Seed文件格式错误：所有seed必须是字符串")
    
    print(f"加载了 {len(seeds)} 个 seed 数据")
    
    # 创建数据合成系统
    synthesizer = GenericDataSynthesis(config=config, output_dir=args.output_dir)
    
    # 运行合成pipeline
    qas = synthesizer.run(seeds)
    
    # 保存结果（trajectories和统计信息，QA对已实时保存）
    synthesizer.save_results()
    
    print(f"\n✅ 全部完成! 共生成 {len(qas)} 个QA对")


if __name__ == "__main__":
    main()

