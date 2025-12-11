# src/mcp_server/system_tools.py
import os
import json
import httpx
import importlib
import logging
import asyncio
from typing import Any
from mcp_server.core.registry import ToolRegistry

# 从环境变量获取 API 地址
RESOURCE_API_URL = os.environ.get("RESOURCE_API_URL", "http://localhost:8000")

logger = logging.getLogger("SystemTools")

# 注册为 "system_resource" 组
@ToolRegistry.register_tool("system_resource",hidden=True)
async def allocate_batch_resources(worker_id: str, resource_types: list[str], timeout: int = 600) -> str:
    """[System Tool] Atomically allocate a batch of resources."""
    if not resource_types:
        return json.dumps({"status": "error", "message": "resource_types cannot be empty"})

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                f"{RESOURCE_API_URL}/allocate",
                json={
                    "worker_id": worker_id, 
                    "resource_types": resource_types,
                    "timeout": timeout
                },
                timeout=timeout + 5
            )
            resp.raise_for_status()
            return json.dumps(resp.json())
            
        except httpx.TimeoutException:
            return json.dumps({"status": "error", "message": "Allocation request timed out"})
        except Exception as e:
            return json.dumps({"status": "error", "message": f"System Allocation Failed: {str(e)}"})

@ToolRegistry.register_tool("system_resource",hidden=True)
async def allocate_single_resource(worker_id: str, resource_type: str, timeout: int = 600) -> str:
    """[System Tool] Allocate a single resource."""
    if not resource_type:
        return json.dumps({"status": "error", "message": "resource_type cannot be empty"})
    return await allocate_batch_resources(worker_id, [resource_type], timeout)

@ToolRegistry.register_tool("system_resource", hidden=True)
async def release_batch_resources(worker_id: str, resource_ids: list[str]) -> str:
    """[System Tool] Batch release resources and trigger local cleanup."""
    if not resource_ids:
        return json.dumps({"status": "success", "message": "No resources to release"})

    results = {}
    async with httpx.AsyncClient() as client:
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
    
    # 清理 Gateway 侧的全局会话缓存
    await _cleanup_resource_sessions(worker_id)
    
    return json.dumps({"status": "completed", "details": results})

@ToolRegistry.register_tool("system_resource", hidden=True)
async def get_batch_initial_observations(worker_id: str) -> str:
    """[System Tool] Retrieve initial observations from local session controllers."""
    observations = {}
    
    vm_module_map = {
        "vm_pyautogui": "mcp_server.vm_pyautogui_server",
        "vm_computer_13": "mcp_server.vm_computer_13_server"
    }

    for res_type, module_name in vm_module_map.items():
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, "GLOBAL_SESSIONS"):
                sessions = getattr(module, "GLOBAL_SESSIONS")
                session = sessions.get(worker_id)
                
                if session and "controller" in session:
                    controller = session["controller"]
                    screenshot = controller.get_screenshot()
                    screenshot_b64 = None
                    if screenshot:
                        import base64
                        screenshot_b64 = base64.b64encode(screenshot).decode('utf-8')
                    a11y_tree = controller.get_accessibility_tree()
                    
                    if screenshot_b64 or a11y_tree:
                        observations[res_type] = {
                            "screenshot": screenshot_b64,
                            "accessibility_tree": a11y_tree
                        }
                    else:
                        observations[res_type] = None
        except Exception as e:
            observations[res_type] = {"error": str(e)}

    try:
        from mcp_server.rag_server import GLOBAL_SESSIONS as RAG_SESSIONS
        if worker_id in RAG_SESSIONS:
            observations["rag"] = {"status": "ready", "message": "RAG session active"}
    except ImportError:
        pass

    return json.dumps(observations)

@ToolRegistry.register_tool("system_resource", hidden=True)
async def setup_batch_resources(worker_id: str, resource_init_configs: dict, allocated_resources: dict = {}) -> str:
    """[System Tool] Dynamically initialize resources."""
    results = {}
    overall_success = True

    if allocated_resources:
        await _sync_resource_sessions(worker_id, allocated_resources)

    if not resource_init_configs:
        return json.dumps({"status": "success", "details": "No config provided, session sync completed"})

    for res_type, config_wrapper in resource_init_configs.items():
        content = config_wrapper.get("content") if isinstance(config_wrapper, dict) else config_wrapper
        msg = "Skipped (No content)"
        success = True

        module_name = f"mcp_server.{res_type}_server"
        try:
            module = importlib.import_module(module_name)
            func_name = f"{res_type}_initialization"
            init_func = getattr(module, func_name, None)

            if init_func:
                logger.info(f"[{worker_id}] Invoking {func_name}...")
                result = init_func(worker_id, content)
                if hasattr(result, '__await__'):
                    result = await result

                if not result:
                    success = False
                    msg = f"{res_type} Init Failed"
                else:
                    msg = f"{res_type} Init Success"
            else:
                msg = "No init logic defined (Success)"

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
    """[Internal] Sync allocated resource info to modules."""
    # RAG Sync
    rag_keys = ["rag", "rag_hybrid"]
    for key in rag_keys:
        if key in allocated_resources:
            rag_info = allocated_resources[key]
            try:
                from mcp_server.rag_server import GLOBAL_SESSIONS as RAG_SESSIONS
                if rag_info.get("id"):
                    RAG_SESSIONS[worker_id] = {
                        "resource_id": rag_info.get("id"),
                        "token": rag_info.get("token"),
                        "base_url": rag_info.get("base_url"),
                        "config_top_k": None
                    }
            except ImportError:
                pass

    # VM Sync
    vm_module_map = {
        "vm_pyautogui": "mcp_server.vm_pyautogui_server",
        "vm_computer_13": "mcp_server.vm_computer_13_server"
    }

    for res_type, res_info in allocated_resources.items():
        if res_type in vm_module_map:
            module_name = vm_module_map[res_type]
            if res_info.get("ip"):
                try:
                    module = importlib.import_module(module_name)
                    if hasattr(module, "GLOBAL_SESSIONS"):
                        target_sessions = getattr(module, "GLOBAL_SESSIONS")
                        from src.utils.desktop_env.controllers.python import PythonController
                        controller = PythonController(vm_ip=res_info["ip"], server_port=res_info.get("port", 5000))
                        target_sessions[worker_id] = {
                            "controller": controller,
                            "env_id": res_info.get("id"),
                            "task_id": "batch_allocated"
                        }
                except Exception as e:
                    logger.error(f"[{worker_id}] VM sync failed for {res_type}: {e}")

async def _cleanup_resource_sessions(worker_id: str):
    """
    [Internal] Cleanup gateway-side sessions by calling module hooks.
    """
    resource_map = {
        "rag": "mcp_server.rag_server",
        "vm_pyautogui": "mcp_server.vm_pyautogui_server",
        "vm_computer_13": "mcp_server.vm_computer_13_server"
    }

    for res_type, module_name in resource_map.items():
        try:
            module = importlib.import_module(module_name)
            
            # 1. Try calling standard cleanup hook
            cleanup_func_name = f"{res_type}_cleanup"
            if hasattr(module, cleanup_func_name):
                cleanup_func = getattr(module, cleanup_func_name)
                if asyncio.iscoroutinefunction(cleanup_func):
                    await cleanup_func(worker_id)
                else:
                    cleanup_func(worker_id)
                logger.info(f"[{worker_id}] Cleaned up {module_name} via hook")
            
            # 2. Fallback: Direct deletion
            elif hasattr(module, "GLOBAL_SESSIONS"):
                sessions = getattr(module, "GLOBAL_SESSIONS")
                if worker_id in sessions:
                    del sessions[worker_id]
                    logger.warning(f"[{worker_id}] Fallback cleanup for {module_name}")

        except ImportError:
            pass
        except Exception as e:
            logger.error(f"[{worker_id}] Cleanup failed for {module_name}: {e}")