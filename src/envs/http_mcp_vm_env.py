import logging
from typing import Dict, Any

from .http_mcp_env import HttpMCPEnv

logger = logging.getLogger("HttpMCPVmEnv")


class HttpMCPVmEnv(HttpMCPEnv):
    """
    MCP 环境子类：面向 VM/桌面资源，启动后会主动获取初始观察。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._has_vm_resources = any(
            isinstance(r, str) and ("vm_" in r or "desktop" in r)
            for r in self.active_resources
        )

    def get_inital_obs(self) -> Dict[str, Any]:
        """调用 MCP 获取初始观察，并应用黑名单过滤（仅 VM/桌面资源）。"""
        combined_obs: Dict[str, Any] = {}
        self.initial_observation = None  # 重置主观察

        resource_blacklist = self.config.get("observation_blacklist", [])
        content_blacklist = self.config.get("observation_content_blacklist", {})

        try:
            res = self._call_tool_sync("get_batch_initial_observations", {"worker_id": self.worker_id})
            data = self._parse_mcp_response(res)

            if isinstance(data, dict) and "error" not in data:
                for resource_type, obs_content in data.items():
                    if resource_type in resource_blacklist:
                        continue

                    filtered_obs_content = obs_content
                    if resource_type in content_blacklist and isinstance(obs_content, dict):
                        filtered_obs_content = obs_content.copy()
                        keys_to_remove = content_blacklist[resource_type]
                        for key in keys_to_remove:
                            if key in filtered_obs_content:
                                del filtered_obs_content[key]

                    combined_obs[resource_type] = filtered_obs_content

                    # 动态设置主观察（优先 VM/桌面且包含截图/a11y/text）
                    if self.initial_observation is None and ("vm" in resource_type.lower() or "desktop" in resource_type.lower()):
                        if filtered_obs_content and isinstance(filtered_obs_content, dict):
                            if any(key in filtered_obs_content for key in ["screenshot", "accessibility_tree", "text"]):
                                self.initial_observation = filtered_obs_content
            else:
                logger.warning(f"[{self.worker_id}] Failed to fetch obs: {data.get('error') if isinstance(data, dict) else data}")

            return combined_obs
        except Exception as e:
            logger.error(f"[{self.worker_id}] Obs fetch error: {e}")
            return combined_obs

    def fetch_initial_observations(self) -> Dict[str, Any]:
        """仅在包含 VM/桌面资源时执行初始观察抓取。"""
        if not getattr(self, "_has_vm_resources", False):
            return {}
        return self.get_inital_obs()
