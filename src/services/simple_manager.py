# src/services/simple_manager.py
import logging
import time
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
        """
        初始化资源池。
        """
        logger.info("Initializing VMPoolImpl directly...")
        try:
            self.pool = VMPoolImpl(**self.config)
            # 使用并行初始化，并发度设为 10（可根据云厂商限制调整）
            success = self.pool.initialize_pool(max_workers=10)
            if success:
                logger.info(f"Resource Pool Initialized. Total: {self.pool.num_items}")
            return success
        except Exception as e:
            logger.error(f"Error during pool initialization: {e}", exc_info=True)
            return False

    def allocate(self, worker_id: str, timeout: float = 60.0) -> Dict[str, Any]:
        """
        申请资源。
        如果当前无可用资源，会阻塞等待直到超时。
        """
        if not self.pool:
            raise RuntimeError("Pool not initialized")

        # 1. 幂等性检查：如果该 worker 已经持有资源，直接返回缓存的连接信息
        with self.pool.pool_lock:
            for entry in self.pool.pool.values():
                if entry.allocated_to == worker_id and entry.status == ResourceStatus.OCCUPIED:
                    logger.info(f"Worker [{worker_id}] already owns {entry.resource_id}. Returning cached info.")
                    return self.pool._get_connection_info(entry)

        logger.info(f"Worker [{worker_id}] requesting resource (Timeout: {timeout}s)...")
        
        # 2. 轮询等待机制 (Polling Wait)
        # 解决 N-Worker > M-VM 时的资源竞争问题
        start_time = time.time()
        
        while True:
            try:
                # 尝试申请资源 (内部 timeout 设为较短，仅用于单次锁竞争)
                resource = self.pool.allocate(worker_id, timeout=1.0)
                
                if resource:
                    self.tracker.record_instance_task(resource['id'], worker_id)
                    logger.info(f"Allocated {resource['id']} to {worker_id}")
                    return resource
            except RuntimeError:
                # 捕获 "No resources available" 异常，继续等待
                pass
            except Exception as e:
                logger.error(f"Unexpected error during allocation: {e}")
                raise e

            # 检查总超时
            elapsed = time.time() - start_time
            if elapsed > timeout:
                break
            
            # 等待一小会儿再次检查 (避免 CPU 空转)
            time.sleep(1.0)

        # 超时未获取到资源
        raise RuntimeError(f"No resources available for {worker_id} after {timeout}s")

    def release(self, resource_id: str, worker_id: str) -> None:
        """
        释放资源。
        通常伴随着 VM 的快照回滚 (reset=True)。
        """
        if not self.pool:
            return
        logger.info(f"Releasing {resource_id} from {worker_id}...")
        try:
            self.pool.release(resource_id, worker_id, reset=True)
            self.tracker.record_instance_cleaned(resource_id)
        except Exception as e:
            logger.error(f"Error releasing resource {resource_id}: {e}")

    def get_status(self) -> Dict[str, Any]:
        """
        获取当前资源池状态监控数据。
        """
        if not self.pool:
            return {"status": "uninitialized"}
        return self.pool.get_stats()