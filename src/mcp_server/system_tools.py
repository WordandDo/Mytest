# src/mcp_server/system_tools.py
import os
import json
import httpx
import importlib
import logging
from mcp_server.core.registry import ToolRegistry

# 从环境变量获取 API 地址
RESOURCE_API_URL = os.environ.get("RESOURCE_API_URL", "http://localhost:8000")

logger = logging.getLogger("SystemTools")

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

# [新增] 注册获取初始 Observation 的工具
@ToolRegistry.register_tool("system_resource")
async def get_batch_initial_observations(worker_id: str) -> str:
    """
    [系统工具] 获取当前已分配给 Worker 的所有资源的初始观测数据。
    返回 JSON 格式的字典，Key 为资源类型（如 'vm', 'rag'），Value 为观测数据或 null。
    """
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                f"{RESOURCE_API_URL}/get_initial_observations",
                json={"worker_id": worker_id},
                timeout=10.0 # 给 VM 截图留出足够时间
            )
            resp.raise_for_status()
            data = resp.json()
            
            if data.get("status") == "success":
                return json.dumps(data.get("observations", {}))
            else:
                # API 返回错误状态
                return json.dumps({"error": data.get("message", "Unknown error from Resource API")})
                
        except Exception as e:
            # 网络或系统异常，返回空 JSON
            return json.dumps({"error": f"System Tool Failed: {str(e)}"})

# [修改] 注册通用资源初始化工具
@ToolRegistry.register_tool("system_resource")
async def setup_batch_resources(worker_id: str, resource_init_configs: dict) -> str:
    """
    [通用系统工具] 动态初始化资源。
    自动根据 res_type 寻找 mcp_server.{res_type}_server 模块下的初始化函数。
    无需为新资源修改此代码。
    """
    results = {}
    overall_success = True

    if not resource_init_configs:
        return json.dumps({"status": "success", "details": "No config provided"})

    for res_type, config_wrapper in resource_init_configs.items():
        # 兼容两种格式: 直接传 content 或者 {"content": ...}
        content = config_wrapper.get("content") if isinstance(config_wrapper, dict) else config_wrapper
        
        # 默认消息
        msg = "Skipped (No content)"
        success = True

        # === 动态加载逻辑 ===
        try:
            # 1. 尝试动态导入模块 (例如: mcp_server.web_server)
            module_name = f"mcp_server.{res_type}_server"
            module = importlib.import_module(module_name)

            # 2. 尝试获取初始化函数 (例如: web_initialization)
            func_name = f"{res_type}_initialization"
            init_func = getattr(module, func_name, None)

            if init_func:
                logger.info(f"[{worker_id}] Invoking {func_name}...")
                # 3. 执行初始化
                # 注意：content 为 None 时也传进去，由函数内部决定是否跳过
                result = init_func(worker_id, content)
                # 处理协程函数
                if hasattr(result, '__await__'):
                    result = await result
                    
                if not result:
                    success = False
                    msg = f"{res_type} Init Failed"
                else:
                    msg = f"{res_type} Init Success"
            else:
                # 模块存在但没定义初始化函数，视为无需初始化，成功
                logger.debug(f"[{worker_id}] No init function found for {res_type}, skipping.")
                msg = "No init logic defined (Success)"

        except ImportError:
            logger.warning(f"[{worker_id}] Module {module_name} not found. Assuming no init needed.")
            msg = "Module not found (Skipped)"
        except Exception as e:
            logger.error(f"[{worker_id}] Error initializing {res_type}: {e}", exc_info=True)
            success = False
            msg = f"Error: {str(e)}"

        results[res_type] = {"success": success, "message": msg}
        if not success:
            overall_success = False

    return json.dumps({
        "status": "success" if overall_success else "partial_error",
        "details": results
    })