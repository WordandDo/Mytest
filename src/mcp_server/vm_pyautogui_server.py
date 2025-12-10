# src/mcp_server/vm_pyautogui_server.py
import sys
import os
import base64
import json
import httpx
import asyncio
import logging
import time
from typing import Optional, List, Any, Union, Callable
from dotenv import load_dotenv

# 引入 MCP 标准类型
from mcp.types import TextContent, ImageContent
load_dotenv()
cwd = os.getcwd()
sys.path.append(cwd)
sys.path.append(os.path.join(cwd, "src"))

from mcp.server.fastmcp import FastMCP
from src.utils.desktop_env.controllers.python import PythonController
# [关键新增] 引入核心 Setup 执行器
from src.utils.desktop_env.controllers.setup import execute_setup_steps

# 导入注册表
from mcp_server.core.registry import ToolRegistry

# 设置日志
logger = logging.getLogger("VMPyAutoGUIServer")

# 设置服务器名称为资源专属名称
mcp = FastMCP("VM PyAutoGUI Gateway")
RESOURCE_API_URL = os.environ.get("RESOURCE_API_URL", "http://localhost:8000")

print(f"🚀 Starting VM PyAutoGUI MCP Server (Registry Mode)")

# 全局会话字典，Key 为 worker_id
GLOBAL_SESSIONS = {}

# =============================================================================
# 1. 核心共享逻辑 (Shared Core Logic)
# =============================================================================

async def _initialize_vm_session(worker_id: str, controller: PythonController, config_data: Any, task_id: str = "unknown") -> bool:
    """
    [Core Logic] 统一的 VM 会话初始化逻辑。
    负责解析配置、执行 Setup 步骤 (下载/安装/执行) 以及加载评估器。
    供 Batch Hook 和 Standalone Tool 复用。
    """
    try:
        # 1. 归一化配置数据
        task_spec = {}
        if isinstance(config_data, dict):
            task_spec = config_data
        elif isinstance(config_data, str) and config_data.strip():
            try:
                if config_data.strip().startswith("{"):
                    task_spec = json.loads(config_data)
                else:
                    # 兼容纯 Python 脚本字符串
                    logger.info(f"[{worker_id}] Executing raw python script...")
                    controller.execute_python_command(config_data)
                    return True
            except json.JSONDecodeError:
                logger.error(f"[{worker_id}] Config string is not valid JSON")
                return False
        else:
            # 空配置直接返回成功
            return True

        # 2. 执行 Benchmark 标准初始化步骤 (Setup Steps)
        setup_steps = task_spec.get("config", [])
        if setup_steps:
            logger.info(f"[{worker_id}] Executing {len(setup_steps)} setup steps via SetupController...")
            # 调用 setup.py 中的强力配置逻辑
            success = execute_setup_steps(controller, setup_steps)
            if not success:
                logger.error(f"[{worker_id}] Setup steps execution failed")
                return False

        # 3. 缓存评估器 (Evaluator)
        evaluator = task_spec.get("evaluator", {})
        if evaluator:
            if worker_id in GLOBAL_SESSIONS:
                GLOBAL_SESSIONS[worker_id]["evaluator"] = evaluator
                logger.info(f"[{worker_id}] Evaluator configuration loaded")

        return True

    except Exception as e:
        logger.error(f"[{worker_id}] Session initialization failed: {e}", exc_info=True)
        return False

async def _cleanup_vm_session_local(worker_id: str):
    """
    [Core Logic] 统一的本地状态清理逻辑。
    只负责从内存中移除 Session，不负责调用 API 释放资源。
    """
    if worker_id in GLOBAL_SESSIONS:
        # 如果需要关闭 socket 连接等操作，可以在这里做
        # session = GLOBAL_SESSIONS[worker_id]
        # if "controller" in session: session["controller"].close()
        
        del GLOBAL_SESSIONS[worker_id]
        logger.info(f"[{worker_id}] VM Session local state cleaned up.")

def _get_controller(worker_id: str) -> PythonController:
    session = GLOBAL_SESSIONS.get(worker_id)
    if not session or not session.get("controller"):
        raise RuntimeError(f"Session not found for worker: {worker_id}. Call 'allocate_single_resource' or 'setup_pyautogui_session' first.")
    return session["controller"]

async def _execute_and_capture(worker_id: str, action_logic: Callable) -> List[Union[TextContent, ImageContent]]:
    """
    Execute the action logic and immediately capture the current screen state and A11y Tree.
    Return a multimodal content list compliant with the MCP protocol.
    """
    contents = []
    
    # 1. 执行动作
    try:
        # 调用传入的 lambda 或函数执行具体的 controller 操作
        action_result = action_logic() 
        # 动作执行成功的文本反馈
        feedback_text = f"Action Executed Successfully."
        if action_result:
             feedback_text += f" Output: {action_result}"
        contents.append(TextContent(type="text", text=feedback_text))
        
    except Exception as e:
        # 如果动作执行失败，返回错误文本
        return [TextContent(type="text", text=f"Error executing action: {str(e)}")]

    # 2. 捕获观测 (Action-as-Observation)
    try:
        ctrl = _get_controller(worker_id)
        
        # A. 获取截图
        screenshot = ctrl.get_screenshot()
        if screenshot:
            screenshot_b64 = base64.b64encode(screenshot).decode('utf-8')
            contents.append(ImageContent(
                type="image",
                data=screenshot_b64,
                mimeType="image/png"
            ))
        
        # B. 获取 A11y Tree
        tree = ctrl.get_accessibility_tree()
        if tree:
            contents.append(TextContent(
                type="text", 
                text=f"<accessibility_tree>\n{tree}\n</accessibility_tree>"
            ))
            
    except Exception as e:
        contents.append(TextContent(type="text", text=f"Warning: Failed to capture post-action observation: {e}"))

    return contents

# =============================================================================
# 2. 批处理钩子 (Batch Hooks) - 供 system_tools 调用
# =============================================================================

async def vm_pyautogui_initialization(worker_id: str, config_content = None) -> bool:
    """
    [Setup Hook] 被 system_tools.setup_batch_resources 调用。
    假设资源已由 system_tools 分配并注入 GLOBAL_SESSIONS。
    """
    session = GLOBAL_SESSIONS.get(worker_id)
    if not session or "controller" not in session:
        logger.error(f"[{worker_id}] No active session found. Cannot initialize.")
        return False

    # 调用核心逻辑
    return await _initialize_vm_session(
        worker_id=worker_id,
        controller=session["controller"],
        config_data=config_content,
        task_id=session.get("task_id", "batch_task")
    )

async def vm_pyautogui_cleanup(worker_id: str):
    """
    [Teardown Hook] 被 system_tools.release_batch_resources (或 cleanup) 调用。
    """
    await _cleanup_vm_session_local(worker_id)

# =============================================================================
# 3. 独立工具 (Standalone Tools) - 供 Agent 直接调用
# =============================================================================

@ToolRegistry.register_tool("pyautogui_lifecycle", hidden=True)
async def setup_pyautogui_session(config_name: str, task_id: str, worker_id: str, init_script: str = "") -> str:
    """
    [System Tool] Initialize VM PyAutoGUI session (Standalone Mode).
    Allocates VM resources and initializes the controller.
    """
    target_resource_type = "vm_pyautogui"
    req_timeout = 600.0 

    # 1. 申请资源
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                f"{RESOURCE_API_URL}/allocate",
                json={
                    "worker_id": worker_id, 
                    "type": target_resource_type,
                    "timeout": req_timeout        
                },
                timeout=req_timeout + 5 
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.TimeoutException:
            return json.dumps({"status": "error", "message": f"Resource queue timeout for {target_resource_type}"})
        except Exception as e:
            return json.dumps({"status": "error", "message": f"Allocation failed: {str(e)}"})

    env_id = data.get("id")
    ip = data.get("ip")
    port = data.get("port", 5000)

    try:
        # 2. 初始化控制器
        controller = PythonController(vm_ip=ip, server_port=port)
        
        # 存入全局会话
        GLOBAL_SESSIONS[worker_id] = {
            "controller": controller,
            "env_id": env_id,
            "task_id": task_id
        }
        
        # 3. 调用核心初始化逻辑
        if init_script:
            await _initialize_vm_session(worker_id, controller, init_script, task_id)
        
        # 4. 获取初始状态
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

@ToolRegistry.register_tool("pyautogui_lifecycle", hidden=True)
async def teardown_pyautogui_environment(worker_id: str) -> str:
    """
    [System Tool] Teardown PyAutoGUI environment.
    Releases resources and cleans up local session.
    """
    # 1. 释放远程资源
    session = GLOBAL_SESSIONS.get(worker_id)
    if session:
        env_id = session.get("env_id")
        async with httpx.AsyncClient() as client:
            try:
                await client.post(f"{RESOURCE_API_URL}/release", 
                                json={"resource_id": env_id, "worker_id": worker_id}, timeout=10)
            except Exception as e:
                logger.error(f"Remote release failed: {e}")

    # 2. 调用核心清理逻辑
    await _cleanup_vm_session_local(worker_id)
    return "Released"

@ToolRegistry.register_tool("pyautogui_lifecycle", hidden=True)
async def evaluate_pyautogui_task(worker_id: str) -> str:
    """
    [System Tool] Evaluate PyAutoGUI task result.
    Calculates the score based on the evaluator configuration.
    """
    session = GLOBAL_SESSIONS.get(worker_id)
    
    if not session or not session.get("evaluator"):
        return "0.0"
    
    evaluator_config = session.get("evaluator", {})
    if not evaluator_config or not isinstance(evaluator_config, dict):
        return "0.0"
    
    # TODO: 实现具体的评估逻辑
    return "0.0"

# --- 观察工具 (Group: desktop_observation) ---

@ToolRegistry.register_tool("pyautogui_observation", hidden=True)
async def start_pyautogui_recording(worker_id: str) -> str:
    """Start screen recording for PyAutoGUI."""
    try:
        ctrl = _get_controller(worker_id)
        ctrl.start_recording()
        return "Recording started"
    except Exception as e:
        return f"Failed to start recording: {str(e)}"

@ToolRegistry.register_tool("pyautogui_observation", hidden=True)
async def stop_pyautogui_recording(worker_id: str, save_path: str) -> str:
    """Stop recording and save file for PyAutoGUI."""
    try:
        ctrl = _get_controller(worker_id)
        # Ensure directory exists
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        ctrl.end_recording(save_path)
        return f"Recording saved to {save_path}"
    except Exception as e:
        return f"Failed to stop recording: {str(e)}"


# =============================================================================
# PyAutoGUI 专属动作 (Group: desktop_action_pyautogui)
# =============================================================================

@ToolRegistry.register_tool("desktop_action_pyautogui")
async def desktop_execute_python_script(worker_id: str, script: str) -> list:
    """Execute a Python script in the desktop environment."""
    ctrl = _get_controller(worker_id)
    return await _execute_and_capture(worker_id, lambda: 
        ctrl.execute_python_command(script)
    )

@ToolRegistry.register_tool("desktop_action_pyautogui")
async def desktop_mouse_button(worker_id: str, action: str, button: str = "left") -> list:
    """Press down or release the mouse button."""
    ctrl = _get_controller(worker_id)
    act_type = "MOUSE_DOWN" if action.lower() == "down" else "MOUSE_UP"
    return await _execute_and_capture(worker_id, lambda: 
        ctrl.execute_action({"action_type": act_type, "parameters": {"button": button}})
    )

@ToolRegistry.register_tool("desktop_action_pyautogui")
async def desktop_control(worker_id: str, action: str) -> list:
    """Execute a control action."""
    ctrl = _get_controller(worker_id)
    act_str = action.upper()
    return await _execute_and_capture(worker_id, lambda: 
        ctrl.execute_action(act_str)
    )