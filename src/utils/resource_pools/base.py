# src/utils/resource_pools/base.py
# -*- coding: utf-8 -*-
import logging
import threading
import time
import os
import concurrent.futures  # [新增]
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue, Empty
from typing import Any, Dict, Optional, List

logger = logging.getLogger(__name__)

class ResourceStatus(Enum):
    """通用资源状态"""
    FREE = "free"
    OCCUPIED = "occupied"
    INITIALIZING = "initializing"
    ERROR = "error"
    STOPPED = "stopped"

@dataclass
class ResourceEntry:
    """通用资源条目基类"""
    resource_id: str
    status: ResourceStatus = ResourceStatus.FREE
    allocated_to: Optional[str] = None
    allocated_at: Optional[float] = None
    error_message: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self):
        if isinstance(self.status, str):
            self.status = ResourceStatus(self.status)

class AbstractPoolManager(ABC):
    """抽象资源池管理器"""

    def __init__(self, num_items: int):
        self.num_items = num_items
        self.pool: Dict[str, ResourceEntry] = {}
        self.free_queue: Queue = Queue()
        self.pool_lock = threading.RLock()
        self.stats = {
            "total": 0, "free": 0, "occupied": 0,
            "error": 0, "allocations": 0, "releases": 0,
        }

        # [第3层超时] 资源占用超时配置
        # 从环境变量读取，默认900秒（15分钟）
        self.max_occupation_time = float(
            os.environ.get("RESOURCE_MAX_OCCUPATION_TIME", "900")
        )

        logger.info(
            f"{self.__class__.__name__} initialized with {num_items} items, "
            f"max_occupation_time={self.max_occupation_time}s"
        )

    @abstractmethod
    def _create_resource(self, index: int) -> ResourceEntry:
        pass

    @abstractmethod
    def _validate_resource(self, entry: ResourceEntry) -> bool:
        pass

    @abstractmethod
    def _get_connection_info(self, entry: ResourceEntry) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _reset_resource(self, entry: ResourceEntry) -> None:
        pass

    @abstractmethod
    def _stop_resource(self, entry: ResourceEntry) -> None:
        pass

    def initialize_pool(self, max_workers: int = 10) -> bool:
        """
        [修改版] 并行初始化资源池
        """
        logger.info(f"Initializing pool with {self.num_items} resources (Parallel, workers={max_workers})...")
        
        success_count = 0
        
        def init_single_resource(i):
            try:
                entry = self._create_resource(i)
                with self.pool_lock:
                    self.pool[entry.resource_id] = entry
                    if entry.status == ResourceStatus.FREE:
                        self.free_queue.put(entry.resource_id)
                        self.stats["free"] += 1
                        self.stats["total"] += 1
                        return True
                    else:
                        self.stats["error"] += 1
                        self.stats["total"] += 1
                        return False
            except Exception as e:
                logger.error(f"Failed to create resource index {i}: {e}", exc_info=True)
                with self.pool_lock:
                    self.stats["error"] += 1
                    self.stats["total"] += 1
                return False

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(init_single_resource, i) for i in range(self.num_items)]
            for future in concurrent.futures.as_completed(futures):
                if future.result():
                    success_count += 1
        
        logger.info(f"Pool initialization completed: {success_count}/{self.num_items} ready")
        return success_count == self.num_items

    def allocate(self, worker_id: str, timeout: float = 30.0) -> Optional[Dict[str, Any]]:
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    resource_id = self.free_queue.get(timeout=1.0)
                    with self.pool_lock:
                        entry = self.pool.get(resource_id)
                        if not entry: continue
                        if entry.status != ResourceStatus.FREE: continue
                        
                        if not self._validate_resource(entry):
                            logger.warning(f"Resource {resource_id} invalid during allocation.")
                            entry.status = ResourceStatus.ERROR
                            self.stats["free"] -= 1
                            self.stats["error"] += 1
                            continue

                        entry.status = ResourceStatus.OCCUPIED
                        entry.allocated_to = worker_id
                        entry.allocated_at = time.time()
                        
                        self.stats["free"] -= 1
                        self.stats["occupied"] += 1
                        self.stats["allocations"] += 1

                        result = self._get_connection_info(entry)
                        if "id" not in result:
                            result["id"] = resource_id
                            
                        # [修改] 增强日志：显示当前资源池剩余情况
                        logger.info(f"✅ Allocated {resource_id} to {worker_id} (Pool Free: {self.stats['free']}/{self.num_items})")
                        return result

                except Empty:
                    continue
                except Exception as exc:
                    logger.error(f"Error allocating resource: {exc}", exc_info=True)
                    continue
            return None

    def release(self, resource_id: str, worker_id: str, reset: bool = True) -> bool:
        with self.pool_lock:
            entry = self.pool.get(resource_id)
            if not entry:
                logger.warning(f"Resource {resource_id} not found for release")
                return False
            if entry.allocated_to != worker_id:
                logger.warning(f"Resource {resource_id} owned by {entry.allocated_to}, {worker_id} tried to release. Ignored.")
                return False

            # 计算占用时长
            duration = time.time() - (entry.allocated_at or time.time())

            if reset:
                try:
                    self._reset_resource(entry)
                except Exception as e:
                    logger.error(f"Failed to reset resource {resource_id}: {e}")

            entry.status = ResourceStatus.FREE
            entry.allocated_to = None
            entry.allocated_at = None
            
            self.stats["occupied"] -= 1
            self.stats["free"] += 1
            self.stats["releases"] += 1
            
            self.free_queue.put(resource_id)
            # [修改] 增强日志：显示占用时长和回收后状态
            logger.info(f"♻️ Released {resource_id} from {worker_id} (Used: {duration:.1f}s, Pool Free: {self.stats['free']})")
            return True
    def stop_all(self) -> None:
        logger.info("Stopping all resources...")
        with self.pool_lock:
            for rid, entry in self.pool.items():
                try:
                    self._stop_resource(entry)
                    entry.status = ResourceStatus.STOPPED
                except Exception as e:
                    logger.error(f"Failed to stop {rid}: {e}")

    def get_stats(self) -> Dict[str, Any]:
        with self.pool_lock:
            stats = self.stats.copy()
            stats["statuses"] = {rid: entry.status.value for rid, entry in self.pool.items()}
            return stats

    # [新增] 获取资源的观测数据接口
    def get_observation(self, resource_id: str) -> Optional[Any]:
        """
        尝试获取资源的当前观测数据。
        默认返回 None，子类可覆盖此方法以提供具体实现 (如 VM 截图)。
        """
        return None

    # [第3层超时] 资源占用超时保护机制
    def check_and_reclaim_timeout_resources(self) -> List[Dict[str, Any]]:
        """
        检查并强制回收超时占用的资源

        遍历所有被占用的资源，如果占用时间超过max_occupation_time，
        则强制释放资源并记录异常日志。

        Returns:
            被回收的资源列表，包含资源ID、占用者、占用时长等信息
        """
        reclaimed = []
        current_time = time.time()

        with self.pool_lock:
            for resource_id, entry in list(self.pool.items()):
                # 只检查占用状态的资源
                if entry.status != ResourceStatus.OCCUPIED:
                    continue

                # 检查是否有分配时间记录
                if not entry.allocated_at:
                    continue

                # 计算占用时长
                occupation_time = current_time - entry.allocated_at

                # 如果超过最大占用时间，强制回收
                if occupation_time > self.max_occupation_time:
                    worker_id = entry.allocated_to or "unknown"

                    logger.error(
                        f"🚨 [ResourceTimeout] Force reclaiming {resource_id} from {worker_id} "
                        f"after {occupation_time:.1f}s (limit: {self.max_occupation_time}s)"
                    )

                    # 尝试重置资源
                    try:
                        self._reset_resource(entry)
                    except Exception as e:
                        logger.error(f"Failed to reset timeout resource {resource_id}: {e}")

                    # 更新资源状态
                    entry.status = ResourceStatus.FREE
                    entry.allocated_to = None
                    entry.allocated_at = None

                    # 更新统计
                    self.stats["occupied"] -= 1
                    self.stats["free"] += 1
                    self.stats["releases"] += 1

                    # 放回空闲队列
                    self.free_queue.put(resource_id)

                    # 记录回收信息
                    reclaimed.append({
                        "resource_id": resource_id,
                        "worker_id": worker_id,
                        "occupation_time": occupation_time,
                        "max_allowed": self.max_occupation_time,
                    })

                    logger.info(
                        f"♻️ [ForcedRelease] {resource_id} reclaimed "
                        f"(was occupied by {worker_id} for {occupation_time:.1f}s)"
                    )

        return reclaimed
