# src/services/simple_manager.py
import logging
from typing import Dict, Any, Optional
from utils.resource_pools.vm_pool import VMPoolImpl
from utils.instance_tracker import get_instance_tracker

logger = logging.getLogger(__name__)

class SimplifiedResourceManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pool: Optional[VMPoolImpl] = None
        self.tracker = get_instance_tracker()

    def initialize(self) -> bool:
        logger.info("Initializing VMPoolImpl directly...")
        try:
            # [关键] 使用 **self.config 将字典解包为关键字参数
            # 这样 provider_name, region 等都会作为独立参数传入 VMPoolImpl.__init__
            self.pool = VMPoolImpl(**self.config)
            
            success = self.pool.initialize_pool()
            if success:
                logger.info(f"Resource Pool Initialized. Total: {self.pool.num_items}")
            return success
        except Exception as e:
            logger.error(f"Error during pool initialization: {e}", exc_info=True)
            return False

    def allocate(self, worker_id: str, timeout: float = 60.0) -> Dict[str, Any]:
        """
        分配资源
        """
        if not self.pool:
            raise RuntimeError("Pool not initialized")

        logger.info(f"Worker [{worker_id}] requesting resource...")
        
        # 调用 VMPoolImpl 的 allocate
        # 注意：原本的 allocate 可能会阻塞，但在 FastAPI 中最好不要阻塞太久
        # 如果您的 AbstractPoolManager 用了锁，这里是线程安全的
        resource = self.pool.allocate(worker_id, timeout=timeout)
        
        if resource:
            # 记录到 Tracker (日志/审计)
            self.tracker.record_instance_task(resource['id'], worker_id)
            logger.info(f"Allocated {resource['id']} to {worker_id}")
            return resource
        else:
            raise RuntimeError(f"No resources available for {worker_id} after {timeout}s")

    def release(self, resource_id: str, worker_id: str) -> None:
        """
        释放资源
        """
        if not self.pool:
            return

        logger.info(f"Releasing {resource_id} from {worker_id}...")
        
        # 调用 VMPoolImpl 的 release (通常包含重置/快照恢复逻辑)
        self.pool.release(resource_id, worker_id, reset=True)
        
        # 记录清理
        self.tracker.record_instance_cleaned(resource_id)

    def get_status(self) -> Dict[str, Any]:
        """获取当前资源池状态"""
        if not self.pool:
            return {"status": "uninitialized"}
        
        # AbstractPoolManager 通常有 get_stats()
        return self.pool.get_stats()