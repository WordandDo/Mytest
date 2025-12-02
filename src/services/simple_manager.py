# src/services/simple_manager.py
import logging
import time
import threading
from typing import Dict, Any, Optional, List
from utils.instance_tracker import get_instance_tracker
from utils.resource_pools.base import ResourceStatus
from utils.resource_pools.factory import ResourcePoolFactory

logger = logging.getLogger(__name__)

class GenericResourceManager:
    def __init__(self, full_config: Dict[str, Any]):
        self.full_config = full_config
        self.pools: Dict[str, Any] = {} 
        self.tracker = get_instance_tracker()
        
        # [核心组件] 使用 Condition 实现全局锁和通知机制
        self.state_cond = threading.Condition()

    def initialize(self) -> bool:
        """根据配置动态初始化所有开启的资源池"""
        logger.info("Initializing All Resource Pools...")
        all_success = True
        
        resources_conf = self.full_config.get("resources", {})
        
        for res_type, res_conf in resources_conf.items():
            # 检查是否启用
            if not res_conf.get("enabled", False):
                logger.info(f"Skipping disabled resource: {res_type}")
                continue

            logger.info(f"--> Init Pool: {res_type}")
            try:
                # 1. 使用工厂创建实例 (此时 config 里已经有了 action_space)
                pool_impl = ResourcePoolFactory.create_pool(
                    class_path=res_conf["implementation_class"],
                    config=res_conf["config"]
                )
                
                # 2. 调用初始化方法 (max_workers 可选写入 config，这里暂定默认值)
                # 假设所有 PoolImpl 都继承自 AbstractPoolManager 并有 initialize_pool
                success = pool_impl.initialize_pool(max_workers=5)
                
                if success:
                    self.pools[res_type] = pool_impl
                    logger.info(f"✅ Pool '{res_type}' initialized. Size: {pool_impl.num_items}")
                else:
                    logger.warning(f"⚠️ Pool '{res_type}' failed to initialize fully.")
                    all_success = False
                    
            except Exception as e:
                logger.error(f"❌ Failed to init pool '{res_type}': {e}", exc_info=True)
                all_success = False

        return all_success

    def allocate_atomic(self, worker_id: str, resource_types: List[str], timeout: float = 600.0) -> Dict[str, Any]:
        """
        [复合方案实现]
        1. Ordering: 对请求资源排序，防止死锁。
        2. Global Lock: 使用 Condition 锁住整个检查过程。
        3. Wait/Notify: 资源不足时挂起等待。
        """
        # [策略1] 强制排序 (Resource Ordering)
        req_types = sorted(list(set(resource_types)))
        
        for r_type in req_types:
            if r_type not in self.pools:
                 raise RuntimeError(f"Resource type '{r_type}' not initialized.")

        logger.info(f"🔄 [AtomicAlloc] Worker={worker_id} Requesting (Sorted): {req_types}")
        
        start_time = time.time()
        
        # [策略2] 全局分配锁 (Global Lock)
        with self.state_cond:
            while True:
                # --- 检查阶段 ---
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
                    allocated_batch = {}
                    try:
                        for r_type in req_types:
                            pool = self.pools[r_type]
                            res = pool.allocate(worker_id, timeout=0.01) 
                            if not res:
                                raise RuntimeError(f"Unexpected allocation failure for {r_type}")
                            allocated_batch[r_type] = res
                            
                        res_ids = [r['id'] for r in allocated_batch.values()]
                        for r_type, res in allocated_batch.items():
                            self.tracker.record_instance_task(res['id'], worker_id)
                        
                        logger.info(f"✅ [AtomicAlloc] Worker={worker_id} Acquired: {res_ids}")
                        return allocated_batch
                        
                    except Exception as e:
                        logger.error(f"Critical error during allocation phase: {e}")
                        for r_type, res in allocated_batch.items():
                            self.pools[r_type].release(res['id'], worker_id, reset=False)
                        raise e

                # --- 等待阶段 ---
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    err_msg = f"Atomic allocation timeout for {req_types} after {elapsed:.1f}s. Missing: {unavailable_resource}"
                    logger.error(f"❌ [AtomicTimeout] Worker={worker_id} {err_msg}")
                    raise RuntimeError(err_msg)
                
                logger.info(f"⏳ [AtomicWait] Worker={worker_id} Waiting for {unavailable_resource}... (Elapsed: {elapsed:.1f}s)")
                self.state_cond.wait(timeout=5.0)

    def allocate(self, worker_id: str, timeout: float = 60.0, resource_type: str = None) -> Dict[str, Any]:
        """通用申请资源逻辑（单资源）"""
        if not resource_type:
             raise ValueError("resource_type must be specified")
        res_map = self.allocate_atomic(worker_id, [resource_type], timeout)
        return res_map[resource_type]

    def release(self, resource_id: str, worker_id: str) -> None:
        """通用释放逻辑"""
        released = False
        target_pool = None
        
        # [策略2 & 3] 获取锁进行释放，并发送通知
        with self.state_cond:
            for name, pool in self.pools.items():
                if resource_id in pool.pool:
                    target_pool = name
                    if pool.release(resource_id, worker_id, reset=True):
                        self.tracker.record_instance_cleaned(resource_id)
                        released = True
                        break 
            
            if released:
                logger.info(f"♻️ [Released] Worker={worker_id} released {resource_id} from pool '{target_pool}'")
                # [策略3] 唤醒所有等待的 Worker
                self.state_cond.notify_all()
                logger.debug("🔔 Notified all waiting workers.")
            else:
                logger.warning(f"⚠️ [ReleaseFail] Could not release {resource_id} (not found or not owned by {worker_id})")

    def release_batch(self, resources: Dict[str, Any], worker_id: str) -> None:
        """批量释放由 allocate_atomic 分配的资源"""
        for r_type, res in resources.items():
            if isinstance(res, dict) and 'id' in res:
                self.release(res['id'], worker_id)

    def get_status(self) -> Dict[str, Any]:
        """动态聚合状态（线程安全）"""
        with self.state_cond:
            return {name: pool.get_stats() for name, pool in self.pools.items()}
    
    # [新增] 聚合观测数据的方法
    def get_initial_observations(self, worker_id: str) -> Dict[str, Any]:
        """
        遍历所有 Pool，收集该 Worker 名下所有资源的 Observation。
        """
        results = {}
        # self.pools 是根据 deployment_config.json 初始化生成的
        for res_type, pool in self.pools.items():
            found_entry = None
            
            # 1. 查找 Worker 拥有的资源 ID
            with pool.pool_lock:
                for entry in pool.pool.values():
                    if entry.allocated_to == worker_id:
                        found_entry = entry
                        break
            
            # 2. 获取观测数据 (如果没找到资源，默认为 None)
            obs = None
            if found_entry:
                try:
                    obs = pool.get_observation(found_entry.resource_id)
                except Exception as e:
                    logger.error(f"Error getting observation for {res_type}: {e}")
            
            results[res_type] = obs
            
        return results

    # [修改] top_k 类型改为 Optional[int] = None
    def query_rag(self, resource_id: str, worker_id: str, query: str, top_k: Optional[int] = None) -> str:
        # RAG 特有方法的特殊处理
        rag_pool = self.pools.get("rag")
        if not rag_pool:
            raise RuntimeError("RAG Pool not initialized")
        return rag_pool.process_query(resource_id, worker_id, query, top_k)