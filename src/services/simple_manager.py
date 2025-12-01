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
        避免 Worker A 占有 VM 等 RAG，而 Worker B 占有 RAG 等 VM 的死锁情况。
        """
        # 去重并校验资源池是否存在
        req_types = list(set(resource_types))
        for r_type in req_types:
            if r_type not in self.pools:
                 raise RuntimeError(f"Resource type '{r_type}' not initialized.")

        logger.info(f"Worker [{worker_id}] atomic requesting: {req_types}")
        
        start_time = time.time()
        while True:
            allocated_batch = {}
            success = True
            
            # 1. 尝试按顺序申请所有资源
            # 注意：这里使用非阻塞或短超时尝试
            for r_type in req_types:
                pool = self.pools[r_type]
                
                try:
                    # 尝试快速获取，不等待
                    resource = pool.allocate(worker_id, timeout=0.1)
                    if resource:
                        allocated_batch[r_type] = resource
                    else:
                        success = False
                        break
                except Exception as e:
                    logger.error(f"Error checking {r_type}: {e}")
                    success = False
                    break
            
            # 2. 判断结果
            if success:
                # 全部申请成功
                for r_type, res in allocated_batch.items():
                    self.tracker.record_instance_task(res['id'], worker_id)
                logger.info(f"✅ Worker [{worker_id}] 原子申请成功: {[r['id'] for r in allocated_batch.values()]}")
                return allocated_batch
            else:
                # 3. 失败回滚：释放本次循环中已获取的部分资源
                if allocated_batch:
                    logger.debug(f"Worker [{worker_id}] 原子申请部分失败，回滚释放: {list(allocated_batch.keys())}")
                    for r_type, res in allocated_batch.items():
                        pool = self.pools[r_type]
                        # reset=False 很重要，避免频繁重置虚拟机导致性能下降，仅释放锁即可
                        pool.release(res['id'], worker_id, reset=False)
                
                # 4. 超时检查
                if time.time() - start_time > timeout:
                    raise RuntimeError(f"Atomic allocation timeout for {req_types} after {timeout}s")
                
                # 5. 随机避退，减少竞争冲突
                time.sleep(random.uniform(1.0, 3.0))

    def allocate(self, worker_id: str, timeout: float = 60.0, resource_type: str = "vm") -> Dict[str, Any]:
        """通用申请资源逻辑（单资源，兼容旧接口）"""
        res_map = self.allocate_atomic(worker_id, [resource_type], timeout)
        return res_map[resource_type]

    def release(self, resource_id: str, worker_id: str) -> None:
        """通用释放逻辑：遍历所有池尝试释放"""
        released = False
        for name, pool in self.pools.items():
            if resource_id in pool.pool:
                if pool.release(resource_id, worker_id, reset=True):
                    self.tracker.record_instance_cleaned(resource_id)
                    released = True
                    break # 找到并释放后退出循环
        
        if not released:
            logger.warning(f"Could not release {resource_id} (not found or not owned by {worker_id})")

    def release_batch(self, resources: Dict[str, Any], worker_id: str) -> None:
        """批量释放由 allocate_atomic 分配的资源"""
        for r_type, res in resources.items():
            if isinstance(res, dict) and 'id' in res:
                self.release(res['id'], worker_id)

    def get_status(self) -> Dict[str, Any]:
        """动态聚合状态"""
        return {name: pool.get_stats() for name, pool in self.pools.items()}
    
    # [修改] top_k 类型改为 Optional[int] = None
    def query_rag(self, resource_id: str, worker_id: str, query: str, top_k: Optional[int] = None) -> str:
        # RAG 特有方法的特殊处理
        rag_pool = self.pools.get("rag")
        if not rag_pool:
            raise RuntimeError("RAG Pool not initialized")
        return rag_pool.process_query(resource_id, worker_id, query, top_k)