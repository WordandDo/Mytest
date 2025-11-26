# src/services/simple_manager.py
import logging
from typing import Dict, Any, Optional
from utils.resource_pools.vm_pool import VMPoolImpl
from utils.instance_tracker import get_instance_tracker
from utils.resource_pools.base import ResourceStatus

logger = logging.getLogger(__name__)

class SimplifiedResourceManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pool: Optional[VMPoolImpl] = None
        self.tracker = get_instance_tracker()

    def initialize(self) -> bool:
        logger.info("Initializing VMPoolImpl directly...")
        try:
            self.pool = VMPoolImpl(**self.config)
            # [修改] 使用并行初始化，并发度设为 10（可根据云厂商限制调整）
            success = self.pool.initialize_pool(max_workers=10)
            if success:
                logger.info(f"Resource Pool Initialized. Total: {self.pool.num_items}")
            return success
        except Exception as e:
            logger.error(f"Error during pool initialization: {e}", exc_info=True)
            return False

    def allocate(self, worker_id: str, timeout: float = 60.0) -> Dict[str, Any]:
        if not self.pool:
            raise RuntimeError("Pool not initialized")

        # [新增] 幂等性检查：如果该 worker 已经持有资源，直接返回
        with self.pool.pool_lock:
            for entry in self.pool.pool.values():
                if entry.allocated_to == worker_id and entry.status == ResourceStatus.OCCUPIED:
                    logger.info(f"Worker [{worker_id}] already owns {entry.resource_id}. Returning cached info.")
                    return self.pool._get_connection_info(entry)

        logger.info(f"Worker [{worker_id}] requesting resource...")
        resource = self.pool.allocate(worker_id, timeout=timeout)
        
        if resource:
            self.tracker.record_instance_task(resource['id'], worker_id)
            logger.info(f"Allocated {resource['id']} to {worker_id}")
            return resource
        else:
            raise RuntimeError(f"No resources available for {worker_id} after {timeout}s")

    def release(self, resource_id: str, worker_id: str) -> None:
        if not self.pool:
            return
        logger.info(f"Releasing {resource_id} from {worker_id}...")
        self.pool.release(resource_id, worker_id, reset=True)
        self.tracker.record_instance_cleaned(resource_id)

    def get_status(self) -> Dict[str, Any]:
        if not self.pool:
            return {"status": "uninitialized"}
        return self.pool.get_stats()