# src/services/simple_manager.py
import logging
import time
from typing import Dict, Any, Optional
from utils.instance_tracker import get_instance_tracker
from utils.resource_pools.base import ResourceStatus
from utils.resource_pools.factory import ResourcePoolFactory  # [新增导入]

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

    def allocate(self, worker_id: str, timeout: float = 60.0, resource_type: str = "vm") -> Dict[str, Any]:
        """通用申请资源逻辑"""
        # [修改] 动态查找 Pool
        pool = self.pools.get(resource_type)
        if not pool:
            available = list(self.pools.keys())
            raise RuntimeError(f"Resource type '{resource_type}' not found or not initialized. Available: {available}")

        # 1. 幂等性检查
        with pool.pool_lock:
            for entry in pool.pool.values():
                if entry.allocated_to == worker_id and entry.status == ResourceStatus.OCCUPIED:
                    logger.info(f"Worker [{worker_id}] already owns {resource_type} {entry.resource_id}. Returning cached info.")
                    return pool._get_connection_info(entry)

        logger.info(f"Worker [{worker_id}] requesting {resource_type} resource...")
        
        # 2. 轮询等待机制
        start_time = time.time()
        while True:
            try:
                resource = pool.allocate(worker_id, timeout=1.0)
                if resource:
                    # 记录追踪
                    self.tracker.record_instance_task(resource['id'], worker_id)
                    logger.info(f"Allocated {resource_type} {resource['id']} to {worker_id}")
                    return resource
            except Exception as e:
                logger.error(f"Unexpected error during allocation: {e}")
            
            if time.time() - start_time > timeout:
                break
            time.sleep(1.0)

        raise RuntimeError(f"No {resource_type} resources available for {worker_id} after {timeout}s")

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