# src/mcp_server/vm_computer_13_server.py
import sys
import os
import base64
import json
import httpx
import asyncio
import logging
from typing import Optional, List, Any, Union, Callable
from dotenv import load_dotenv

from mcp.types import TextContent, ImageContent
load_dotenv()
cwd = os.getcwd()
sys.path.append(cwd)
sys.path.append(os.path.join(cwd, "src"))

from mcp.server.fastmcp import FastMCP
from src.utils.desktop_env.controllers.python import PythonController
from src.utils.desktop_env.controllers.setup import execute_setup_steps

from mcp_server.core.registry import ToolRegistry

logger = logging.getLogger("VMComputer13Server")

mcp = FastMCP("VM Computer 13 Gateway")
RESOURCE_API_URL = os.environ.get("RESOURCE_API_URL", "http://localhost:8000")

print(f"🚀 Starting VM Computer 13 MCP Server (Registry Mode)")

GLOBAL_SESSIONS = {}

# =============================================================================
# 1. 核心共享逻辑 (Shared Core Logic)
# =============================================================================

async def _initialize_vm_session(worker_id: str, controller: PythonController, config_data: Any, task_id: str = "unknown") -> bool:
    """[Core Logic] 统一 VM 会话初始化逻辑"""
    try:
        task_spec = {}
        if isinstance(config_data, dict):
            task_spec = config_data
        elif isinstance(config_data, str) and config_data.strip():
            try:
                if config_data.strip().startswith("{"):
                    task_spec = json.loads(config_data)
                else:
                    logger.info(f"[{worker_id}] Executing raw python script...")
                    controller.execute_python_command(config_data)
                    return True
            except json.JSONDecodeError:
                logger.error(f"[{worker_id}] Config string is not valid JSON")
                return False
        else:
            return True

        setup_steps = task_spec.get("config", [])
        if setup_steps:
            logger.info(f"[{worker_id}] Executing {len(setup_steps)} setup steps...")
            success = execute_setup_steps(controller, setup_steps)
            if not success:
                logger.error(f"[{worker_id}] Setup steps execution failed")
                return False

        evaluator = task_spec.get("evaluator", {})
        if evaluator:
            if worker_id in GLOBAL_SESSIONS:
                GLOBAL_SESSIONS[worker_id]["evaluator"] = evaluator

        return True

    except Exception as e:
        logger.error(f"[{worker_id}] Session initialization failed: {e}", exc_info=True)
        return False

async def _cleanup_vm_session_local(worker_id: str):
    """[Core Logic] 统一本地清理逻辑"""
    if worker_id in GLOBAL_SESSIONS:
        del GLOBAL_SESSIONS[worker_id]
        logger.info(f"[{worker_id}] VM Session local state cleaned up.")

def _get_controller(worker_id: str) -> PythonController:
    session = GLOBAL_SESSIONS.get(worker_id)
    if not session or not session.get("controller"):
        raise RuntimeError(f"Session not found for worker: {worker_id}")
    return session["controller"]

async def _execute_and_capture(worker_id: str, action_logic: Callable) -> List[Union[TextContent, ImageContent]]:
    """Execute action and capture observation"""
    contents = []
    try:
        action_result = action_logic() 
        feedback_text = f"Action Executed Successfully."
        if action_result:
             feedback_text += f" Output: {action_result}"
        contents.append(TextContent(type="text", text=feedback_text))
    except Exception as e:
        return [TextContent(type="text", text=f"Error executing action: {str(e)}")]

    try:
        ctrl = _get_controller(worker_id)
        screenshot = ctrl.get_screenshot()
        if screenshot:
            screenshot_b64 = base64.b64encode(screenshot).decode('utf-8')
            contents.append(ImageContent(type="image", data=screenshot_b64, mimeType="image/png"))
        
        tree = ctrl.get_accessibility_tree()
        if tree:
            contents.append(TextContent(type="text", text=f"<accessibility_tree>\n{tree}\n</accessibility_tree>"))
    except Exception as e:
        contents.append(TextContent(type="text", text=f"Warning: Failed to capture post-action observation: {e}"))

    return contents

# =============================================================================
# 2. 批处理钩子 (Batch Hooks)
# =============================================================================

async def vm_computer_13_initialization(worker_id: str, config_content = None) -> bool:
    """[Setup Hook] 被 system_tools 调用"""
    session = GLOBAL_SESSIONS.get(worker_id)
    if not session or "controller" not in session:
        logger.error(f"[{worker_id}] No active session found.")
        return False

    return await _initialize_vm_session(
        worker_id=worker_id,
        controller=session["controller"],
        config_data=config_content,
        task_id=session.get("task_id", "batch_task")
    )

async def vm_computer_13_cleanup(worker_id: str):
    """[Teardown Hook] 被 system_tools 调用"""
    await _cleanup_vm_session_local(worker_id)

# =============================================================================
# 3. 独立工具 (Standalone Tools)
# =============================================================================

@ToolRegistry.register_tool("computer_lifecycle", hidden=True)
async def setup_computer_13_session(config_name: str, task_id: str, worker_id: str, init_script: str = "") -> str:
    """[System Tool] Initialize VM Computer 13 session (Standalone)."""
    target_resource_type = "vm_computer_13"
    req_timeout = 600.0 

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                f"{RESOURCE_API_URL}/allocate",
                json={"worker_id": worker_id, "type": target_resource_type, "timeout": req_timeout},
                timeout=req_timeout + 5 
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return json.dumps({"status": "error", "message": f"Allocation failed: {str(e)}"})

    env_id = data.get("id")
    ip = data.get("ip")
    port = data.get("port", 5000)

    try:
        controller = PythonController(vm_ip=ip, server_port=port)
        GLOBAL_SESSIONS[worker_id] = {
            "controller": controller,
            "env_id": env_id,
            "task_id": task_id
        }
        
        if init_script:
            await _initialize_vm_session(worker_id, controller, init_script, task_id)
        
        screenshot = controller.get_screenshot()
        screenshot_b64 = base64.b64encode(screenshot).decode('utf-8') if screenshot else ""
        
        return json.dumps({
            "status": "success", 
            "observation": {
                "screenshot": screenshot_b64,
                "accessibility_tree": controller.get_accessibility_tree()
            }
        })
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})

@ToolRegistry.register_tool("computer_lifecycle", hidden=True)
async def teardown_computer_13_environment(worker_id: str) -> str:
    """[System Tool] Teardown VM Computer 13 environment."""
    session = GLOBAL_SESSIONS.get(worker_id)
    if session:
        env_id = session.get("env_id")
        async with httpx.AsyncClient() as client:
            try:
                await client.post(f"{RESOURCE_API_URL}/release", 
                                json={"resource_id": env_id, "worker_id": worker_id}, timeout=10)
            except Exception:
                pass

    await _cleanup_vm_session_local(worker_id)
    return "Released"