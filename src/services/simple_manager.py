# src/services/simple_manager.py
import logging
import time
import random
from typing import Dict, Any, Optional, List
from utils.instance_tracker import get_instance_tracker
from utils.resource_pools.base import ResourceStatus
from utils.resource_pools.factory import ResourcePoolFactory

logger = logging.getLogger(__name__)

class GenericResourceManager:  # [建议重命名类，或保留原名]
    def __init__(self, full_config: Dict[str, Any]):
        self.full_config = full_config
        # [修改] 统一存储所有 Pool: {"vm": pool_obj, "rag": pool_obj}
        self.pools: Dict[str, Any] = {} 
        self.tracker = get_instance_tracker()

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
                # 1. 使用工厂创建实例
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
        原子化申请多种资源。要么全部成功，要么全部失败并重试。
        """
        req_types = list(set(resource_types))
        for r_type in req_types:
            if r_type not in self.pools:
                 raise RuntimeError(f"Resource type '{r_type}' not initialized.")

        logger.info(f"🔄 [AtomicAlloc] Worker={worker_id} Start requesting: {req_types}")
        
        start_time = time.time()
        attempt_count = 0
        
        while True:
            attempt_count += 1
            allocated_batch = {}
            success = True

            for r_type in req_types:
                pool = self.pools[r_type]
                try:
                    # 尝试快速获取，不等待 (timeout=0.1 避免长时间阻塞)
                    resource = pool.allocate(worker_id, timeout=0.1)
                    if resource:
                        allocated_batch[r_type] = resource
                    else:
                        success = False
                        # [Log] 拿单个资源失败
                        logger.warning(f"⚠️ [AtomicAlloc] Worker={worker_id} failed to get '{r_type}' in attempt #{attempt_count}")
                        break
                except Exception as e:
                    logger.error(f"Error checking {r_type}: {e}")
                    success = False
                    break
            
            if success:
                # [Log] 全部成功
                res_ids = [r['id'] for r in allocated_batch.values()]
                for r_type, res in allocated_batch.items():
                    self.tracker.record_instance_task(res['id'], worker_id)
                logger.info(f"✅ [AtomicAlloc] Worker={worker_id} Success. IDs={res_ids}")
                return allocated_batch
            else:
                # 失败回滚
                if allocated_batch:
                    acquired_keys = list(allocated_batch.keys())
                    logger.warning(f"⏪ [AtomicRollback] Worker={worker_id} rolling back: {acquired_keys}")
                    for r_type, res in allocated_batch.items():
                        pool = self.pools[r_type]
                        pool.release(res['id'], worker_id, reset=False)
                
                # 超时检查
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    err_msg = f"Atomic allocation timeout for {req_types} after {elapsed:.1f}s"
                    logger.error(f"❌ [AtomicTimeout] Worker={worker_id} {err_msg}")
                    raise RuntimeError(err_msg)
                
                # 随机避退
                sleep_time = random.uniform(2.0, 5.0) # [调整] 稍微增加等待时间，减少日志刷屏
                logger.info(f"⏳ [AtomicWait] Worker={worker_id} Waiting {sleep_time:.1f}s before retry... (Elapsed: {elapsed:.1f}s)")
                time.sleep(sleep_time)

    def allocate(self, worker_id: str, timeout: float = 60.0, resource_type: str = "vm") -> Dict[str, Any]:
        """通用申请资源逻辑（单资源，兼容旧接口）"""
        res_map = self.allocate_atomic(worker_id, [resource_type], timeout)
        return res_map[resource_type]

    def release(self, resource_id: str, worker_id: str) -> None:
        """通用释放逻辑"""
        released = False
        target_pool = None
        
        for name, pool in self.pools.items():
            if resource_id in pool.pool:
                target_pool = name
                if pool.release(resource_id, worker_id, reset=True):
                    self.tracker.record_instance_cleaned(resource_id)
                    released = True
                    break 
        
        if released:
            logger.info(f"♻️ [Released] Worker={worker_id} released {resource_id} from pool '{target_pool}'")
        else:
            logger.warning(f"⚠️ [ReleaseFail] Could not release {resource_id} (not found or not owned by {worker_id})")

    def release_batch(self, resources: Dict[str, Any], worker_id: str) -> None:
        """批量释放由 allocate_atomic 分配的资源"""
        for r_type, res in resources.items():
            if isinstance(res, dict) and 'id' in res:
                self.release(res['id'], worker_id)

    def get_status(self) -> Dict[str, Any]:
        """动态聚合状态"""
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