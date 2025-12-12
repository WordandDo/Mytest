# src/utils/resource_pools/vm_pool.py
# -*- coding: utf-8 -*-
import logging
import time
import requests
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from utils.desktop_env.providers import create_vm_manager_and_provider
from utils.resource_pools.base import AbstractPoolManager, ResourceEntry, ResourceStatus

logger = logging.getLogger(__name__)
_aliyun_snapshot_lock = threading.Lock()


@dataclass
class VMResourceEntry(ResourceEntry):
    """VM 特有的资源条目"""
    ip: Optional[str] = None
    port: int = 5000
    chromium_port: int = 9222
    vnc_port: int = 8006
    vlc_port: int = 8080
    path_to_vm: Optional[str] = None
    snapshot_id: Optional[str] = None


class VMPoolImpl(AbstractPoolManager):
    """
    VM 池资源管理器实现 (Server 端)
    继承 AbstractPoolManager，实现 VM 具体逻辑。
    """
    def __init__(
        self,
        num_vms: int = 5,
        provider_name: str = "aliyun",
        region: Optional[str] = None,
        path_to_vm: Optional[str] = None,
        snapshot_name: str = "init_state",
        action_space: str = "computer_13",
        screen_size: Tuple[int, int] = (1920, 1080),
        headless: bool = True,
        require_a11y_tree: bool = True,
        require_terminal: bool = False,
        os_type: str = "Ubuntu",
        client_password: str = "password",
        **kwargs,
    ):
        # 初始化基类
        super().__init__(num_items=num_vms)
        
        # 保存 VM 特定配置
        self.provider_name = provider_name
        self.region = region
        self.path_to_vm_template = path_to_vm
        self.snapshot_name = snapshot_name
        self.action_space = action_space
        self.screen_size = tuple(screen_size) if screen_size else (1920, 1080)
        self.headless = headless
        self.require_a11y_tree = require_a11y_tree
        self.require_terminal = require_terminal
        self.os_type = os_type
        self.client_password = client_password
        self.extra_kwargs = kwargs

    def _create_resource(self, index: int) -> VMResourceEntry:
        """实现 VM 创建逻辑"""
        vm_id = f"vm_{index + 1}"
        logger.info("Initializing %s...", vm_id)
        
        manager, provider = create_vm_manager_and_provider(
            self.provider_name, self.region or "", use_proxy=False
        )

        # 准备配置
        desktop_env_kwargs = {
            "provider_name": self.provider_name,
            "region": self.region,
            "path_to_vm": self.path_to_vm_template,
            "snapshot_name": self.snapshot_name,
            "action_space": self.action_space,
            "screen_size": self.screen_size,
            "headless": self.headless,
            "require_a11y_tree": self.require_a11y_tree,
            "require_terminal": self.require_terminal,
            "os_type": self.os_type,
            "client_password": self.client_password,
            **self.extra_kwargs,
        }

        # 解析 VM 路径
        if self.path_to_vm_template:
            vm_path = self.path_to_vm_template
        else:
            vm_path = manager.get_vm_path(
                os_type=self.os_type,
                region=self.region or "",
                screen_size=self.screen_size,
            )
        
        if not vm_path:
            raise RuntimeError(f"Failed to resolve vm_path for {vm_id}")

        # 启动 VM
        provider.start_emulator(vm_path, self.headless, self.os_type)
        
        # 获取 IP
        vm_ip_ports = provider.get_ip_address(vm_path).split(":")
        vm_ip = vm_ip_ports[0]
        
        # 构造 Entry
        entry = VMResourceEntry(
            resource_id=vm_id,
            status=ResourceStatus.FREE,
            ip=vm_ip,
            port=int(vm_ip_ports[1]) if len(vm_ip_ports) > 1 else 5000,
            chromium_port=int(vm_ip_ports[2]) if len(vm_ip_ports) > 2 else 9222,
            vnc_port=int(vm_ip_ports[3]) if len(vm_ip_ports) > 3 else 8006,
            vlc_port=int(vm_ip_ports[4]) if len(vm_ip_ports) > 4 else 8080,
            path_to_vm=vm_path,
            config=desktop_env_kwargs
        )

        # 为新实例捕获“纯净”快照（在 _initialize_vm_session 之前）
        clean_snapshot_id = self._capture_clean_snapshot(provider, vm_path, vm_id)
        if clean_snapshot_id:
            entry.snapshot_id = clean_snapshot_id
            logger.info("✓ %s clean snapshot captured: %s", vm_id, clean_snapshot_id)
        elif self.provider_name == "aliyun":
            logger.warning("%s clean snapshot capture skipped/failed; fast rollback will fall back to recreate.", vm_id)

        # 向上层传递合适的 snapshot_name：Aliyun 用捕获的 ID，其它 provider 用配置值（VMware 需要）
        snapshot_for_config = None
        if self.provider_name == "aliyun":
            snapshot_for_config = clean_snapshot_id  # 仅当捕获成功才传递
        else:
            snapshot_for_config = self.snapshot_name
        if snapshot_for_config:
            entry.config["snapshot_name"] = snapshot_for_config
        elif "snapshot_name" in entry.config:
            entry.config.pop("snapshot_name", None)

        logger.info("✓ %s initialized: ip=%s", vm_id, vm_ip)
        return entry

    def _validate_resource(self, entry: ResourceEntry) -> bool:
        # 确保是 VM Entry 且 IP 存在
        if not isinstance(entry, VMResourceEntry): return False
        if entry.ip is None:
            logger.warning(f"VM {entry.resource_id} has no IP")
            return False
        return True

    def _get_connection_info(self, entry: ResourceEntry) -> Dict[str, Any]:
        # 返回 VM 连接信息
        assert isinstance(entry, VMResourceEntry)
        return {
            "id": entry.resource_id,
            "ip": entry.ip,
            "port": entry.port,
            "chromium_port": entry.chromium_port,
            "vnc_port": entry.vnc_port,
            "vlc_port": entry.vlc_port,
            "path_to_vm": entry.path_to_vm,
            "config": entry.config,
        }

    def _reset_resource(self, entry: ResourceEntry) -> None:
        # Revert VM Snapshot
        assert isinstance(entry, VMResourceEntry)
        if not entry.path_to_vm: return

        _, provider = create_vm_manager_and_provider(
            self.provider_name, self.region or "", use_proxy=False
        )

        is_aliyun = self.provider_name == "aliyun"
        snapshot_ref = entry.snapshot_id if is_aliyun else (entry.snapshot_id or self.snapshot_name)
        new_vm_path = provider.revert_to_snapshot(entry.path_to_vm, snapshot_ref)
        recreated = bool(new_vm_path and new_vm_path != entry.path_to_vm)
        if recreated:
            logger.info(f"VM {entry.resource_id} path changed: {entry.path_to_vm} -> {new_vm_path}")
            entry.path_to_vm = new_vm_path

        # 如果实例被重建或尚未绑定快照，则为当前系统盘补拍“纯净快照”
        if recreated or entry.snapshot_id is None:
            updated_snapshot = self._capture_clean_snapshot(provider, entry.path_to_vm, entry.resource_id)
            if updated_snapshot:
                entry.snapshot_id = updated_snapshot
                logger.info("Refreshed clean snapshot for %s: %s", entry.resource_id, updated_snapshot)
            elif is_aliyun:
                logger.warning("Aliyun clean snapshot refresh failed for %s; will rely on recreate if ResetDisk unavailable.", entry.resource_id)

        # 更新传递给上层的 snapshot_name（VMware 需要显式名称；Aliyun 传递捕获到的 ID）
        snapshot_for_config = None
        if is_aliyun:
            snapshot_for_config = entry.snapshot_id
        else:
            snapshot_for_config = self.snapshot_name
        if snapshot_for_config:
            entry.config["snapshot_name"] = snapshot_for_config
        elif "snapshot_name" in entry.config:
            entry.config.pop("snapshot_name", None)

        # 重置后更新 IP
        vm_ip_ports = provider.get_ip_address(entry.path_to_vm).split(":")
        new_ip = vm_ip_ports[0]
        if new_ip != entry.ip:
            logger.info(f"VM {entry.resource_id} IP changed: {entry.ip} -> {new_ip}")
            entry.ip = new_ip
            if len(vm_ip_ports) > 1:
                entry.port = int(vm_ip_ports[1])
                # ... update other ports if needed

    def _stop_resource(self, entry: ResourceEntry) -> None:
        assert isinstance(entry, VMResourceEntry)
        if not entry.path_to_vm: return
        _, provider = create_vm_manager_and_provider(
            self.provider_name, self.region or "", use_proxy=False
        )
        provider.stop_emulator(entry.path_to_vm)

    # ------------------------------------------------------------------ #
    # Aliyun 专用：为当前系统盘捕获纯净快照（用于 ResetDisk 快速回滚）
    # ------------------------------------------------------------------ #
    def _capture_clean_snapshot(self, provider, instance_id: str, vm_id: str) -> Optional[str]:
        """
        在 VM 刚创建完成且未运行 _initialize_vm_session 前拍摄一次快照，
        保留“纯净”状态用于后续 ResetDisk。仅对 Aliyun 生效，其他厂商直接跳过。
        """
        if self.provider_name != "aliyun":
            return None

        try:
            from utils.desktop_env.providers.aliyun.provider import AliyunProvider
            from alibabacloud_ecs20140526 import models as ecs_models  # type: ignore
            from alibabacloud_tea_util.client import Client as TeaClient  # type: ignore
        except Exception as exc:
            logger.warning("Aliyun snapshot dependencies not available, skip clean snapshot: %s", exc)
            return None

        if not isinstance(provider, AliyunProvider):
            return None

        try:
            with _aliyun_snapshot_lock:
                disk_id = provider._get_system_disk_id(instance_id)
                snapshot_req = ecs_models.CreateSnapshotRequest(
                    disk_id=disk_id,
                    snapshot_name=f"clean-{vm_id}-{int(time.time())}",
                    description="vm_pool clean snapshot before _initialize_vm_session"
                )
                resp = provider.client.create_snapshot(snapshot_req)
                snapshot_id = getattr(resp.body, "snapshot_id", None)
                if not snapshot_id:
                    logger.warning("Aliyun returned empty snapshot id for %s", instance_id)
                    return None

            self._wait_snapshot_ready(provider, snapshot_id, TeaClient)
            return snapshot_id
        except Exception as exc:
            logger.warning("Failed to capture clean snapshot for %s: %s", instance_id, exc)
            return None

    def _wait_snapshot_ready(self, provider, snapshot_id: str, tea_client_cls, timeout: int = 600, interval: int = 5) -> None:
        """
        等待快照变为可用状态。若超时则抛出异常，让上层决定是否继续。
        """
        try:
            from alibabacloud_ecs20140526 import models as ecs_models  # type: ignore
        except Exception as exc:
            logger.warning("Aliyun snapshot wait skipped due to missing dependency: %s", exc)
            return

        start = time.time()
        while True:
            if time.time() - start > timeout:
                raise TimeoutError(f"Snapshot {snapshot_id} not ready within {timeout}s")

            try:
                req = ecs_models.DescribeSnapshotsRequest(
                    region_id=getattr(provider, "region", None),
                    snapshot_ids=tea_client_cls.to_jsonstring([snapshot_id])
                )
                resp = provider.client.describe_snapshots(req)
                snapshots = resp.body.snapshots.snapshot
                if snapshots:
                    status = snapshots[0].status
                    if status.lower() == "accomplished":
                        logger.info("Snapshot %s is ready", snapshot_id)
                        return
                    logger.info("Snapshot %s status=%s, waiting...", snapshot_id, status)
            except Exception as exc:
                logger.warning("Error while waiting snapshot %s: %s", snapshot_id, exc)

            time.sleep(interval)

    # [新增] 实现获取 VM 观测数据
    # [已移除] get_observation 方法。现在由 MCP Gateway 服务直接与 VM 的 DesktopEnv Agent 通信以获取观测数据。
    # 此更改简化了 VMPoolImpl，并将观测逻辑集中到 MCP 层。
