from typing import Any# src/mcp_server/system_tools.py
import os
import json
import httpx
import importlib
import logging
from typing import Any
from mcp_server.core.registry import ToolRegistry

# 从环境变量获取 API 地址
RESOURCE_API_URL = os.environ.get("RESOURCE_API_URL", "http://localhost:8000")

logger = logging.getLogger("SystemTools")

# 注册为 "system_resource" 组
@ToolRegistry.register_tool("system_resource")
async def allocate_batch_resources(worker_id: str, resource_types: list[str], timeout: int = 600) -> str:
    """
    [System Tool] Atomically allocate a batch of resources.
    Communicates directly with the Resource API to ensure all resources are allocated together or none at all, preventing deadlocks.
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

# [新增] 统一的批量资源释放工具
@ToolRegistry.register_tool("system_resource", hidden=True)
async def release_batch_resources(worker_id: str, resource_ids: list[str]) -> str:
    """
    [System Tool] Batch release resources.
    Accepts a list of resource IDs and releases them one by one via the Resource API.
    """
    if not resource_ids:
        return json.dumps({"status": "success", "message": "No resources to release"})

    results = {}
    async with httpx.AsyncClient() as client:
        # 并行释放可能导致后端竞争，此处暂时采用安全串行释放
        for rid in resource_ids:
            try:
                resp = await client.post(
                    f"{RESOURCE_API_URL}/release",
                    json={"worker_id": worker_id, "resource_id": rid},
                    timeout=10.0
                )
                if resp.status_code == 200:
                    results[rid] = "released"
                else:
                    results[rid] = f"failed: {resp.status_code}"
            except Exception as e:
                logger.error(f"Failed to release resource {rid}: {e}")
                results[rid] = f"error: {str(e)}"
    
    # 清理 Gateway 侧的全局会话缓存 (Important!)
    await _cleanup_resource_sessions(worker_id)
    
    return json.dumps({"status": "completed", "details": results})

# [修改] 注册获取初始 Observation 的工具
@ToolRegistry.register_tool("system_resource")
async def get_batch_initial_observations(worker_id: str) -> str:
    """
    [System Tool] Retrieve initial observations directly from local session controllers.
    """
    observations = {}
    
    # 定义资源类型到模块的映射 (与 _sync_resource_sessions 中保持一致)
    vm_module_map = {
        "vm_pyautogui": "mcp_server.vm_pyautogui_server",
        "vm_computer_13": "mcp_server.vm_computer_13_server"
    }

    # 1. 尝试从 VM 模块获取截图
    for res_type, module_name in vm_module_map.items():
        try:
            module = importlib.import_module(module_name)
            # 检查该模块是否有会话记录
            if hasattr(module, "GLOBAL_SESSIONS"):
                sessions = getattr(module, "GLOBAL_SESSIONS")
                session = sessions.get(worker_id)
                
                if session and "controller" in session:
                    controller = session["controller"]
                    
                    # === 获取截图 ===
                    screenshot = controller.get_screenshot()
                    screenshot_b64 = None
                    if screenshot:
                        import base64
                        screenshot_b64 = base64.b64encode(screenshot).decode('utf-8')
                    
                    # === 获取 Accessibility Tree ===
                    a11y_tree = controller.get_accessibility_tree()
                    
                    # 构造返回数据
                    if screenshot_b64 or a11y_tree:
                        observations[res_type] = {
                            "screenshot": screenshot_b64,
                            "accessibility_tree": a11y_tree,
                            "message": "Observation fetched from local controller"
                        }
                    else:
                        observations[res_type] = None
        except Exception as e:
            logger.error(f"Error fetching obs for {res_type}: {e}")
            observations[res_type] = {"error": str(e)}

    # 2. 尝试获取 RAG 状态 (RAG 通常没有截图，保持原状或返回状态)
    # RAG 的状态通常是静态的，这里可以简单返回 ready 或者去 ResourceAPI 查，
    # 但为了简化，如果 RAG 已分配，我们可以返回一个简单的状态对象
    try:
        from mcp_server.rag_server import RAG_SESSIONS
        if worker_id in RAG_SESSIONS:
            observations["rag"] = {
                "status": "ready", 
                "message": "RAG session active"
            }
    except ImportError:
        pass

    return json.dumps(observations)

# [修改] 注册通用资源初始化工具
@ToolRegistry.register_tool("system_resource", hidden=True)
async def setup_batch_resources(worker_id: str, resource_init_configs: dict, allocated_resources: dict = {}) -> str:
    """
    [System Tool] Dynamically initialize resources.
    Automatically finds the initialization function in 'mcp_server.{res_type}_server' based on the resource type.
    No need to modify this code for new resources.

    Args:
        worker_id: Worker ID
        resource_init_configs: Initialization configuration dictionary
        allocated_resources: Allocated resource information (optional, for state synchronization)
    """
    results = {}
    overall_success = True

    if not resource_init_configs:
        return json.dumps({"status": "success", "details": "No config provided"})

    # 在初始化前，先同步资源状态到各模块的全局变量
    if allocated_resources:
        logger.info(f"[{worker_id}] Syncing allocated resources to module sessions...")
        await _sync_resource_sessions(worker_id, allocated_resources)

    for res_type, config_wrapper in resource_init_configs.items():
        # 兼容两种格式: 直接传 content 或者 {"content": ...}
        content = config_wrapper.get("content") if isinstance(config_wrapper, dict) else config_wrapper

        # 默认消息
        msg = "Skipped (No content)"
        success = True

        # === 动态加载逻辑 ===
        module_name = f"mcp_server.{res_type}_server"
        try:
            # 1. 尝试动态导入模块 (例如: mcp_server.web_server)
            module = importlib.import_module(module_name)

            # 2. 尝试获取初始化函数 (例如: web_initialization)
            func_name = f"{res_type}_initialization"
            init_func = getattr(module, func_name, None)

            if init_func:
                logger.info(f"[{worker_id}] Invoking {func_name}...")
                # 3. 执行初始化
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


async def _sync_resource_sessions(worker_id: str, allocated_resources: dict):
    """
    [内部函数] 将原子分配的资源信息同步到各模块的全局变量
    """
    # 同步 RAG 资源
    if "rag" in allocated_resources:
        rag_info = allocated_resources["rag"]
        try:
            from mcp_server.rag_server import RAG_SESSIONS
            resource_id = rag_info.get("id")
            token = rag_info.get("token")
            # [关键修正] 必须同时提取 base_url
            base_url = rag_info.get("base_url") 
            
            if resource_id and token: # 建议同时检查 base_url
                RAG_SESSIONS[worker_id] = {
                    "resource_id": resource_id, 
                    "token": token,
                    "base_url": base_url, # [新增] 存入直连地址
                    "config_top_k": None  # 初始化配置项，防止后续查询出错
                }
                logger.info(f"[{worker_id}] Synced RAG session (Direct Mode: {base_url})")
        except ImportError:
            pass

    # [修复] 同步 VM 资源
    # 定义资源类型到模块名的映射 (osworld_server 已被移除)
    vm_module_map = {
        "vm_pyautogui": "mcp_server.vm_pyautogui_server",
        "vm_computer_13": "mcp_server.vm_computer_13_server"
    }

    # 遍历分配的资源，查找是否有 VM 类型的资源
    for res_type, res_info in allocated_resources.items():
        # 如果该资源类型在映射表中，则进行同步
        if res_type in vm_module_map:
            module_name = vm_module_map[res_type]
            vm_ip = res_info.get("ip")
            vm_port = res_info.get("port", 5000)
            env_id = res_info.get("id")

            if vm_ip and env_id:
                try:
                    # 动态导入对应的 Server 模块
                    module = importlib.import_module(module_name)
                    
                    # 获取该模块的 GLOBAL_SESSIONS
                    if hasattr(module, "GLOBAL_SESSIONS"):
                        target_sessions = getattr(module, "GLOBAL_SESSIONS")
                        
                        from src.utils.desktop_env.controllers.python import PythonController
                        controller = PythonController(vm_ip=vm_ip, server_port=vm_port)
                        
                        # [关键] 写入目标模块的会话字典
                        target_sessions[worker_id] = {
                            "controller": controller,
                            "env_id": env_id,
                            "task_id": "batch_allocated"
                        }
                        logger.info(f"[{worker_id}] Synced VM session to {module_name}")
                    else:
                        logger.error(f"[{worker_id}] Module {module_name} has no GLOBAL_SESSIONS")
                        
                except Exception as e:
                    logger.error(f"[{worker_id}] VM sync failed for {res_type}: {e}")

async def _cleanup_resource_sessions(worker_id: str):
    """
    [内部函数] 清理 Gateway 侧各模块的全局会话缓存
    """
    # 清理 RAG
    try:
        from mcp_server.rag_server import RAG_SESSIONS
        if worker_id in RAG_SESSIONS:
            del RAG_SESSIONS[worker_id]
            logger.info(f"[{worker_id}] Cleaned up RAG session")
    except ImportError:
        pass

    # 清理所有已知 VM 模块的会话 (已移除 osworld)
    vm_modules = [
        "mcp_server.vm_pyautogui_server",
        "mcp_server.vm_computer_13_server"
    ]
    
    for module_name in vm_modules:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, "GLOBAL_SESSIONS"):
                target_sessions = getattr(module, "GLOBAL_SESSIONS")
                if worker_id in target_sessions:
                    del target_sessions[worker_id]
                    logger.info(f"[{worker_id}] Cleaned up session in {module_name}")
        except ImportError:
            pass