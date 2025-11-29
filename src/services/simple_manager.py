# src/services/simple_manager.py
import logging
import time
from typing import Dict, Any, Optional
from utils.resource_pools.vm_pool import VMPoolImpl
from utils.resource_pools.rag_pool import RAGPoolImpl  # [新增]
from utils.instance_tracker import get_instance_tracker
from utils.resource_pools.base import ResourceStatus

logger = logging.getLogger(__name__)

class SimplifiedResourceManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vm_pool: Optional[VMPoolImpl] = None   # [修改] 重命名
        self.rag_pool: Optional[RAGPoolImpl] = None # [新增]
        self.tracker = get_instance_tracker()

    def initialize(self) -> bool:
        """
        初始化所有资源池。
        """
        logger.info("Initializing Resource Pools...")
        try:
            # 1. 初始化 VM 池
            logger.info("--> Init VM Pool")
            self.vm_pool = VMPoolImpl(**self.config)
            vm_success = self.vm_pool.initialize_pool(max_workers=10)

            # 2. 初始化 RAG 池 [新增]
            logger.info("--> Init RAG Pool")
            # 提取 RAG 相关配置，如果有专门的前缀处理更好，这里假设 config 中包含了所需字段
            self.rag_pool = RAGPoolImpl(**self.config)
            rag_success = self.rag_pool.initialize_pool(max_workers=5)
            
            total_vm = self.vm_pool.num_items if self.vm_pool else 0
            total_rag = self.rag_pool.num_items if self.rag_pool else 0

            if vm_success and rag_success:
                logger.info(f"✅ All Pools Initialized. VMs: {total_vm}, RAGs: {total_rag}")
                return True
            else:
                logger.warning(f"⚠️ Partial initialization. VM Success: {vm_success}, RAG Success: {rag_success}")
                return False
                
        except Exception as e:
            logger.error(f"Error during pool initialization: {e}", exc_info=True)
            return False

    def allocate(self, worker_id: str, timeout: float = 60.0, resource_type: str = "vm") -> Dict[str, Any]:
        """
        申请资源。
        :param resource_type: "vm" 或 "rag"
        """
        # [新增] 根据类型选择资源池
        if resource_type == "rag":
            pool = self.rag_pool
        else:
            pool = self.vm_pool

        if not pool:
            raise RuntimeError(f"Pool for type '{resource_type}' not initialized")

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
                    # 记录追踪（如果是 VM 则记录，RAG 可选）
                    if resource_type == "vm":
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
        """
        释放资源。尝试在两个池中查找并释放。
        """
        released = False
        # 尝试从 VM 池释放
        if self.vm_pool and resource_id in self.vm_pool.pool:
            if self.vm_pool.release(resource_id, worker_id, reset=True):
                self.tracker.record_instance_cleaned(resource_id)
                released = True
        
        # 尝试从 RAG 池释放 [新增]
        if not released and self.rag_pool and resource_id in self.rag_pool.pool:
            self.rag_pool.release(resource_id, worker_id, reset=True)
            released = True

        if not released:
            logger.warning(f"Could not release {resource_id} (not found or not owned by {worker_id})")

    def get_status(self) -> Dict[str, Any]:
        """
        获取合并的状态监控数据。
        """
        status = {}
        if self.vm_pool:
            status["vm"] = self.vm_pool.get_stats()
        if self.rag_pool:
            status["rag"] = self.rag_pool.get_stats()
        return status