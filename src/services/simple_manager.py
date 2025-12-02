# src/services/simple_manager.py
# 通用资源管理器实现，负责具体资源的分配、释放和管理逻辑
# 采用工厂模式和配置驱动的方式，支持动态加载和管理各种资源类型

import logging
import time
import threading
from typing import Dict, Any, Optional, List
from utils.instance_tracker import get_instance_tracker  # 实例跟踪器，用于跟踪资源实例的分配和释放
from utils.resource_pools.base import ResourceStatus  # 资源状态枚举
from utils.resource_pools.factory import ResourcePoolFactory  # 资源池工厂，用于动态创建资源池实例

logger = logging.getLogger(__name__)

class GenericResourceManager:
    """
    通用资源管理器
    
    负责管理各种资源池的初始化、分配、释放和状态监控等功能。
    采用工厂模式和配置驱动，支持动态扩展新的资源类型。
    """
    
    def __init__(self, full_config: Dict[str, Any]):
        """
        初始化资源管理器
        
        Args:
            full_config: 完整的配置字典，包含所有资源类型的配置信息
        """
        self.full_config = full_config
        self.pools: Dict[str, Any] = {}  # 存储已初始化的资源池实例
        self.tracker = get_instance_tracker()  # 获取实例跟踪器实例
        
        # [核心组件] 使用 Condition 实现全局锁和通知机制
        # 用于确保资源分配的原子性和线程安全，防止死锁和资源竞争
        self.state_cond = threading.Condition()

    def initialize(self) -> bool:
        """
        根据配置动态初始化所有开启的资源池
        
        遍历配置中的所有资源类型，对启用的资源类型创建对应的资源池实例。
        
        Returns:
            初始化是否全部成功
        """
        logger.info("Initializing All Resource Pools...")
        all_success = True
        
        # 获取资源配置部分
        resources_conf = self.full_config.get("resources", {})
        
        # 遍历每种资源配置
        for res_type, res_conf in resources_conf.items():
            # 检查是否启用该资源类型
            if not res_conf.get("enabled", False):
                logger.info(f"Skipping disabled resource: {res_type}")
                continue

            logger.info(f"--> Init Pool: {res_type}")
            try:
                # 1. 使用工厂创建实例 (此时 config 里已经有了 action_space)
                # 通过资源池工厂动态创建资源池实例
                pool_impl = ResourcePoolFactory.create_pool(
                    class_path=res_conf["implementation_class"],  # 资源池实现类路径
                    config=res_conf["config"]  # 资源池配置
                )
                
                # 2. 调用初始化方法 (max_workers 可选写入 config，这里暂定默认值)
                # 假设所有 PoolImpl 都继承自 AbstractPoolManager 并有 initialize_pool
                success = pool_impl.initialize_pool(max_workers=5)
                
                if success:
                    # 初始化成功，将资源池实例保存到pools字典中
                    self.pools[res_type] = pool_impl
                    logger.info(f"✅ Pool '{res_type}' initialized. Size: {pool_impl.num_items}")
                else:
                    # 初始化失败
                    logger.warning(f"⚠️ Pool '{res_type}' failed to initialize fully.")
                    all_success = False
                    
            except Exception as e:
                # 捕获初始化过程中的异常
                logger.error(f"❌ Failed to init pool '{res_type}': {e}", exc_info=True)
                all_success = False

        return all_success

    def allocate_atomic(self, worker_id: str, resource_types: List[str], timeout: float = 600.0) -> Dict[str, Any]:
        """
        原子性地分配多个资源
        
        采用复合方案实现资源分配，确保分配过程的原子性和线程安全：
        1. Ordering: 对请求资源排序，防止死锁。
        2. Global Lock: 使用 Condition 锁住整个检查过程。
        3. Wait/Notify: 资源不足时挂起等待。
        
        Args:
            worker_id: 工作节点ID
            resource_types: 需要分配的资源类型列表
            timeout: 分配超时时间（秒）
            
        Returns:
            分配成功的资源信息字典
            
        Raises:
            RuntimeError: 资源分配失败或超时
        """
        # [策略1] 强制排序 (Resource Ordering)
        # 对请求的资源类型进行排序，确保所有线程按照相同顺序申请资源，防止死锁
        req_types = sorted(list(set(resource_types)))
        
        # 检查请求的资源类型是否都已初始化
        for r_type in req_types:
            if r_type not in self.pools:
                 raise RuntimeError(f"Resource type '{r_type}' not initialized.")

        logger.info(f"🔄 [AtomicAlloc] Worker={worker_id} Requesting (Sorted): {req_types}")
        
        start_time = time.time()
        
        # [策略2] 全局分配锁 (Global Lock)
        # 使用with语句确保锁的正确获取和释放
        with self.state_cond:
            while True:
                # --- 检查阶段 ---
                # 检查所有请求的资源是否都有空闲实例
                all_available = True
                unavailable_resource = None
                
                for r_type in req_types:
                    pool = self.pools[r_type]
                    stats = pool.get_stats()
                    if stats['free'] <= 0:
                        all_available = False
                        unavailable_resource = r_type
                        break
                
                # --- 分配阶段 ---
                if all_available:
                    # 所有资源都有空闲实例，开始分配
                    allocated_batch = {}
                    try:
                        # 依次分配每种资源
                        for r_type in req_types:
                            pool = self.pools[r_type]
                            res = pool.allocate(worker_id, timeout=0.01) 
                            if not res:
                                raise RuntimeError(f"Unexpected allocation failure for {r_type}")
                            allocated_batch[r_type] = res
                            
                        # 记录分配的资源ID并跟踪实例
                        res_ids = [r['id'] for r in allocated_batch.values()]
                        for r_type, res in allocated_batch.items():
                            self.tracker.record_instance_task(res['id'], worker_id)
                        
                        logger.info(f"✅ [AtomicAlloc] Worker={worker_id} Acquired: {res_ids}")
                        return allocated_batch
                        
                    except Exception as e:
                        # 分配过程中出现异常，回滚已分配的资源
                        logger.error(f"Critical error during allocation phase: {e}")
                        for r_type, res in allocated_batch.items():
                            # 仅释放锁，不重置资源状态
                            self.pools[r_type].release(res['id'], worker_id, reset=False)
                        raise e

                # --- 等待阶段 ---
                # 计算已等待时间
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    # 等待超时，抛出异常
                    err_msg = f"Atomic allocation timeout for {req_types} after {elapsed:.1f}s. Missing: {unavailable_resource}"
                    logger.error(f"❌ [AtomicTimeout] Worker={worker_id} {err_msg}")
                    raise RuntimeError(err_msg)
                
                # 记录等待信息并挂起线程
                logger.info(f"⏳ [AtomicWait] Worker={worker_id} Waiting for {unavailable_resource}... (Elapsed: {elapsed:.1f}s)")
                # 等待其他线程释放资源，超时时间为5秒
                self.state_cond.wait(timeout=5.0)

    def allocate(self, worker_id: str, timeout: float = 60.0, resource_type: str = None) -> Dict[str, Any]:
        """
        分配单个资源
        
        Args:
            worker_id: 工作节点ID
            timeout: 分配超时时间（秒）
            resource_type: 资源类型
            
        Returns:
            分配成功的资源信息
            
        Raises:
            ValueError: 未指定资源类型
        """
        # 检查资源类型是否指定
        if not resource_type:
             raise ValueError("resource_type must be specified")
        # 调用原子分配方法分配资源
        res_map = self.allocate_atomic(worker_id, [resource_type], timeout)
        return res_map[resource_type]

    def release(self, resource_id: str, worker_id: str) -> None:
        """
        释放资源
        
        Args:
            resource_id: 资源ID
            worker_id: 工作节点ID
        """
        released = False
        target_pool = None
        
        # [策略2 & 3] 获取锁进行释放，并发送通知
        # 使用with语句确保锁的正确获取和释放
        with self.state_cond:
            # 查找资源所属的资源池
            for name, pool in self.pools.items():
                if resource_id in pool.pool:
                    target_pool = name
                    # 调用资源池的释放方法
                    if pool.release(resource_id, worker_id, reset=True):
                        # 释放成功，记录清理事件
                        self.tracker.record_instance_cleaned(resource_id)
                        released = True
                        break 
            
            if released:
                # 释放成功，记录日志并通知其他等待的线程
                logger.info(f"♻️ [Released] Worker={worker_id} released {resource_id} from pool '{target_pool}'")
                # [策略3] 唤醒所有等待的 Worker
                self.state_cond.notify_all()
                logger.debug("🔔 Notified all waiting workers.")
            else:
                # 释放失败，记录警告日志
                logger.warning(f"⚠️ [ReleaseFail] Could not release {resource_id} (not found or not owned by {worker_id})")

    def release_batch(self, resources: Dict[str, Any], worker_id: str) -> None:
        """
        批量释放资源
        
        Args:
            resources: 资源信息字典
            worker_id: 工作节点ID
        """
        # 遍历资源字典，逐一释放每个资源
        for r_type, res in resources.items():
            if isinstance(res, dict) and 'id' in res:
                self.release(res['id'], worker_id)

    def get_status(self) -> Dict[str, Any]:
        """
        获取所有资源池的状态信息
        
        Returns:
            各资源池的状态信息字典
        """
        # 使用全局锁确保状态获取的原子性
        with self.state_cond:
            return {name: pool.get_stats() for name, pool in self.pools.items()}
    
    # [新增] 聚合观测数据的方法
    def get_initial_observations(self, worker_id: str) -> Dict[str, Any]:
        """
        遍历所有 Pool，收集该 Worker 名下所有资源的 Observation。
        
        Args:
            worker_id: 工作节点ID
            
        Returns:
            各资源类型的观测数据字典
        """
        results = {}
        # self.pools 是根据 deployment_config.json 初始化生成的
        for res_type, pool in self.pools.items():
            found_entry = None
            
            # 1. 查找 Worker 拥有的资源 ID
            # 使用资源池的锁确保线程安全
            with pool.pool_lock:
                for entry in pool.pool.values():
                    if entry.allocated_to == worker_id:
                        found_entry = entry
                        break
            
            # 2. 获取观测数据 (如果没找到资源，默认为 None)
            obs = None
            if found_entry:
                try:
                    # 调用资源池的观测方法获取数据
                    obs = pool.get_observation(found_entry.resource_id)
                except Exception as e:
                    logger.error(f"Error getting observation for {res_type}: {e}")
            
            results[res_type] = obs
            
        return results

    # [修改] top_k 类型改为 Optional[int] = None
    def query_rag(self, resource_id: str, worker_id: str, query: str, top_k: Optional[int] = None) -> str:
        """
        RAG查询方法
        
        Args:
            resource_id: 资源ID
            worker_id: 工作节点ID
            query: 查询内容
            top_k: 返回结果数量（可选）
            
        Returns:
            查询结果文本
            
        Raises:
            RuntimeError: RAG资源池未初始化
        """
        # RAG 特有方法的特殊处理
        rag_pool = self.pools.get("rag")
        if not rag_pool:
            raise RuntimeError("RAG Pool not initialized")
        # 调用RAG资源池的查询方法
        return rag_pool.process_query(resource_id, worker_id, query, top_k)