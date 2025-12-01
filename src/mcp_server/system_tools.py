# src/mcp_server/system_tools.py
import os
import json
import httpx
from mcp_server.core.registry import ToolRegistry

# 从环境变量获取 API 地址
RESOURCE_API_URL = os.environ.get("RESOURCE_API_URL", "http://localhost:8000")

# 注册为 "system_resource" 组
@ToolRegistry.register_tool("system_resource")
async def allocate_batch_resources(worker_id: str, resource_types: list[str], timeout: int = 600) -> str:
    """
    [系统工具] 原子化批量申请资源。
    直接与 Resource API 通信，确保多个资源要么全有，要么全无，避免死锁。
    """
    if not resource_types:
        return json.dumps({"status": "error", "message": "resource_types cannot be empty"})

    async with httpx.AsyncClient() as client:
        try:
            # Gateway 代表 Client 向 Resource API 发起 HTTP 请求
            resp = await client.post(
                f"{RESOURCE_API_URL}/allocate",
                json={
                    "worker_id": worker_id, 
                    "resource_types": resource_types, # 传递列表，触发后端原子申请
                    "timeout": timeout
                },
                timeout=timeout + 5 # 客户端超时要比逻辑超时稍长
            )
            resp.raise_for_status()
            # 直接返回 Resource API 的响应（包含连接信息等）
            return json.dumps(resp.json())
            
        except httpx.TimeoutException:
            return json.dumps({"status": "error", "message": "Allocation request timed out"})
        except Exception as e:
            return json.dumps({"status": "error", "message": f"System Allocation Failed: {str(e)}"})