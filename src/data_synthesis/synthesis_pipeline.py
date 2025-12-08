"""
数据合成主Pipeline (修复版)

包含完整的资源生命周期管理：
1. env_start(): 连接 MCP Server
2. allocate_resource(): 锁定后端资源 (RAG Index/VM)
3. run(): 执行任务
4. cleanup(): 释放资源
"""

import json
import os
import bdb
import hashlib
import time
from typing import List, Dict, Set

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 仅导入基础 Environment 类
from envs import Environment
from models import TrajectoryNode, Trajectory, SynthesizedQA
from synthesis_config import SynthesisConfig
from trajectory_sampler import GenericTrajectorySampler
from trajectory_selector import GenericTrajectorySelector
from qa_synthesizer import GenericQASynthesizer


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
        
        # 1. 创建环境
        print(f"初始化 {config.environment_mode.upper()} Environment...")
        self.environment = self._create_environment()
        
        # [关键修复] 启动环境并分配资源
        self._initialize_environment_resources()
        
        # 2. 创建三个组件
        # 注意：Sampler 必须在环境 ready (tools loaded) 后初始化
        self.sampler = GenericTrajectorySampler(
            environment=self.environment,
            config=config
        )
        
        self.selector = GenericTrajectorySelector(config=config)
        
        self.synthesizer = GenericQASynthesizer(config=config)
        
        # 存储结果
        self.trajectory_tree: Dict[str, TrajectoryNode] = {}
        self.selected_trajectories: List[Trajectory] = []
        self.synthesized_qas: List[SynthesizedQA] = []
        
        # 初始化输出文件路径
        self.qa_file_path = None
        self.traj_file_path = None
        
        # 已处理的source_id集合
        self.processed_source_ids: Set[str] = set()
    
    def _initialize_environment_resources(self):
        """[新增] 处理环境连接和资源分配"""
        print("🔗正在连接 MCP Server...")
        
        # 1. 建立连接 (获取工具列表)
        if hasattr(self.environment, "env_start"):
            self.environment.env_start()
            
        # 2. 申请资源 (锁定 RAG 索引或 VM)
        if hasattr(self.environment, "allocate_resource"):
            print("🔐 正在申请后端资源 (Resource Allocation)...")
            # 使用固定 ID，串行模式下无冲突
            success = self.environment.allocate_resource("synthesis_serial_worker")
            if success:
                print("✅ 资源分配成功")
            else:
                print("❌ 资源分配失败! 后端可能未就绪或被占用")
                # 即使失败也尝试继续，可能处于无状态模式
                
        # 3. 稍微等待工具列表同步
        time.sleep(2) 
    
    def _create_environment(self) -> Environment:
        """根据配置创建相应的环境 (按需导入)"""
        mode = self.config.environment_mode.lower()
        kwargs = self.config.environment_kwargs.copy()
        kwargs['model_name'] = self.config.model_name
        
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
                pass 
            from envs import RAGEnvironment
            return RAGEnvironment(**kwargs)
        elif mode == "osworld" or mode == "gui":
            required_params = ['path_to_vm']
            missing = [p for p in required_params if p not in kwargs]
            if missing:
                raise ValueError(f"OSWorld环境需要提供以下参数: {', '.join(missing)}")
            from envs import OSWorldEnvironment
            return OSWorldEnvironment(**kwargs)
        else:
            raise ValueError(f"不支持的环境模式: {mode}")
    
    def _initialize_output_files(self):
        """初始化输出文件路径并创建输出目录"""
        os.makedirs(self.output_dir, exist_ok=True)
        
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
    
    def _generate_source_id(self, seed_data: str, seed_idx: int) -> str:
        content_hash = hashlib.md5(seed_data.encode('utf-8')).hexdigest()[:8]
        return f"src_{seed_idx:04d}_{content_hash}"
    
    def _load_processed_source_ids(self):
        self.processed_source_ids.clear()
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
    
    def _save_qa_immediately(self, qa: SynthesizedQA):
        with open(self.qa_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(qa.to_dict(), ensure_ascii=False) + "\n")
    
    def _save_trajectories_immediately(self, trajectories: List[Trajectory]):
        with open(self.traj_file_path, "a", encoding="utf-8") as f:
            for traj in trajectories:
                f.write(json.dumps(traj.to_dict(), ensure_ascii=False) + "\n")
    
    def cleanup(self):
        """[新增] 清理资源"""
        print("\n🧹 正在清理资源...")
        if hasattr(self.environment, "cleanup"):
            self.environment.cleanup()
    
    def run(self, seeds: List[str]) -> List[SynthesizedQA]:
        """
        运行完整的数据合成pipeline
        """
        if self.config.number_of_seed is not None:
            seeds = seeds[:self.config.number_of_seed]
        
        print(f"\n{'='*80}")
        print(f"🚀 通用Agent数据合成 Pipeline 启动")
        print(f"{'='*80}")
        print(f"环境模式: {self.config.environment_mode}")
        print(f"Seed说明: {self.config.seed_description or '(未指定)'}")
        # 此时工具应该已经加载了
        available_tools = [t['name'] for t in self.sampler.available_tools]
        print(f"可用工具: {available_tools}")
        if not available_tools:
            print("⚠️ 警告: 没有发现任何工具！请检查 Gateway 连接或资源分配状态。")
            
        print(f"总Seed数量: {len(seeds)}")
        print(f"模型: {self.config.model_name}")
        print(f"{'='*80}\n")
        
        self._initialize_output_files()
        
        all_qas = []
        skipped_count = 0
        
        try:
            for seed_idx, seed_data in enumerate(seeds, 1):
                source_id = self._generate_source_id(seed_data, seed_idx)
                
                if source_id in self.processed_source_ids:
                    skipped_count += 1
                    print(f"\n⏭️  跳过 Seed {seed_idx}/{len(seeds)} (已处理: {source_id})")
                    continue
                
                print(f"\n\n{'#'*80}")
                print(f"处理 Seed {seed_idx}/{len(seeds)}")
                print(f"Source ID: {source_id}")
                print(f"内容: {seed_data}")
                print(f"{'#'*80}\n")
                
                try:
                    # Step 1: Trajectory Sampling
                    print(f"\n📊 步骤 1/3: Trajectory Sampling")
                    self.trajectory_tree = self.sampler.sample_trajectory_tree(seed_data)
                    
                    # Step 2: Trajectory Selection
                    print(f"\n🎯 步骤 2/3: Trajectory Selection")
                    self.selected_trajectories = self.selector.select_trajectories(
                        nodes=self.trajectory_tree,
                        root_id=self.sampler.root_id,
                        seed_data=seed_data,
                        source_id=source_id,
                        max_selected_traj=self.config.max_selected_traj
                    )
                    
                    # Step 3: QA Synthesis
                    print(f"\n✨ 步骤 3/3: QA Synthesis")
                    for qa_idx, trajectory in enumerate(self.selected_trajectories):
                        qa = self.synthesizer.synthesize_qa(trajectory, qa_idx)
                        if qa:
                            all_qas.append(qa)
                            self.synthesized_qas.append(qa)
                            self._save_qa_immediately(qa)
                    
                    if self.selected_trajectories:
                        self._save_trajectories_immediately(self.selected_trajectories)
                    
                    print(f"\n✅ Seed {seed_idx} 完成! 生成了 {len([qa for qa in all_qas if qa.source_id == source_id])} 个QA对")
                    
                except Exception as e:
                    if isinstance(e, bdb.BdbQuit):
                        raise e
                    print(f"\n❌ Seed {seed_idx} 失败: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        finally:
            # 确保退出时释放资源
            self.cleanup()
        
        print(f"\n\n{'='*80}")
        print(f"🎉 数据合成完成!")
        print(f"{'='*80}")
        print(f"总Seed数量: {len(seeds)} 个")
        print(f"已跳过: {skipped_count} 个")
        print(f"新处理: {len(seeds) - skipped_count} 个")
        print(f"成功生成: {len(all_qas)} 个QA对")
        print(f"{'='*80}\n")
        
        return all_qas
    
    def save_results(self):
        """显示结果保存位置"""
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
        seeds = [str(s) if not isinstance(s, str) else s for s in seeds]
    
    print(f"加载了 {len(seeds)} 个 seed 数据")
    
    synthesizer = GenericDataSynthesis(config=config, output_dir=args.output_dir)
    
    # 运行合成pipeline
    qas = synthesizer.run(seeds)
    
    synthesizer.save_results()
    
    print(f"\n✅ 全部完成! 共生成 {len(qas)} 个QA对")


if __name__ == "__main__":
    main()