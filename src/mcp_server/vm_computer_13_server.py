# src/mcp_server/vm_computer_13_server.py
import sys
import os
import base64
import json
import httpx
import asyncio
import logging
from typing import Optional, List, Any, Union, Callable, Annotated, Literal
from dotenv import load_dotenv
from pydantic import Field

try:
    from mcp.types import TextContent, ImageContent  # type: ignore
except Exception:
    class TextContent(dict):
        def __init__(self, type: str, text: str):
            super().__init__({"type": type, "text": text})

    class ImageContent(dict):
        def __init__(self, type: str, data: str, mimeType: str):
            super().__init__({"type": type, "data": data, "mimeType": mimeType})
load_dotenv()
cwd = os.getcwd()
sys.path.append(cwd)
sys.path.append(os.path.join(cwd, "src"))

try:
    from mcp.server.fastmcp import FastMCP  # type: ignore
except Exception:
    FastMCP = None  # type: ignore
try:
    from src.utils.desktop_env.controllers.python import PythonController  # type: ignore
except Exception:
    PythonController = None  # type: ignore

try:
    from src.utils.desktop_env.controllers.setup import execute_setup_steps as _execute_setup_steps  # type: ignore
except Exception:
    def _execute_setup_steps(*args, **kwargs):  # type: ignore
        raise ImportError("execute_setup_steps requires optional dependencies (e.g., playwright).")

from mcp_server.core.registry import ToolRegistry

logger = logging.getLogger("VMComputer13Server")

mcp = FastMCP("VM Computer 13 Gateway") if FastMCP else None
RESOURCE_API_URL = os.environ.get("RESOURCE_API_URL", "http://localhost:8000")

print(f"🚀 Starting VM Computer 13 MCP Server (Registry Mode)")

GLOBAL_SESSIONS = {}

# =============================================================================
# 1. 核心共享逻辑 (Shared Core Logic)
# =============================================================================

async def _safe_release_resource(resource_id: str, worker_id: str, reason: str = ""):
    """安全释放资源，用于初始化失败时的回滚。"""
    if not resource_id:
        return
    async with httpx.AsyncClient() as client:
        try:
            await client.post(
                f"{RESOURCE_API_URL}/release",
                json={"resource_id": resource_id, "worker_id": worker_id},
                timeout=10
            )
        except Exception as e:
            logger.error(f"[{worker_id}] Safe release failed for {resource_id} ({reason}): {e}")

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
            success = _execute_setup_steps(controller, setup_steps)
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

    session_created = False
    try:
        controller = PythonController(vm_ip=ip, server_port=port)
        GLOBAL_SESSIONS[worker_id] = {
            "controller": controller,
            "env_id": env_id,
            "task_id": task_id
        }
        session_created = True
        
        if init_script:
            init_ok = await _initialize_vm_session(worker_id, controller, init_script, task_id)
            if not init_ok:
                raise RuntimeError("VM initialization failed")
        
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
        await _safe_release_resource(env_id, worker_id, reason="setup_computer_13_session")
        if session_created:
            await _cleanup_vm_session_local(worker_id)
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

# =============================================================================
# 4. 操作与观察工具 (Action & Observation Tools)
#    Group: desktop_action_computer_13, computer13_observation
# =============================================================================

def _ctrl(worker_id: str):
    return _get_controller(worker_id)


# --- 观察工具 ---
@ToolRegistry.register_tool("computer13_observation", hidden=True)
async def start_computer13_recording(
    worker_id: Annotated[str, Field(description="Worker id/session key (auto-injected by HttpMCPEnv).")]
) -> str:
    """Start screen recording for Computer-13 VM (hidden)."""
    try:
        ctrl = _ctrl(worker_id)
        ctrl.start_recording()
        return "Recording started"
    except Exception as e:
        return f"Failed to start recording: {str(e)}"


@ToolRegistry.register_tool("computer13_observation", hidden=True)
async def stop_computer13_recording(
    worker_id: Annotated[str, Field(description="Worker id/session key (auto-injected by HttpMCPEnv).")],
    save_path: Annotated[str, Field(description="Absolute path to save the recording file.", min_length=1)]
) -> str:
    """Stop recording and save file for Computer-13 VM (hidden)."""
    try:
        ctrl = _ctrl(worker_id)
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        ctrl.end_recording(save_path)
        return f"Recording saved to {save_path}"
    except Exception as e:
        return f"Failed to stop recording: {str(e)}"


# --- 基础动作工具 ---
@ToolRegistry.register_tool("desktop_action_computer_13")
async def computer13_execute_python_script(
    worker_id: Annotated[str, Field(description="Worker id/session key (auto-injected by HttpMCPEnv).")],
    script: Annotated[
        str,
        Field(description="Python statements to run inside the VM. pyautogui is pre-imported; keep scripts short and deterministic.", min_length=1)
    ]
) -> list:
    """
    Execute a Python snippet inside the VM (pyautogui pre-imported, FAILSAFE disabled).
    Include time.sleep() where UI latency is expected; avoid input()/long loops.
    """
    ctrl = _ctrl(worker_id)
    return await _execute_and_capture(worker_id, lambda: ctrl.execute_python_command(script))


@ToolRegistry.register_tool("desktop_action_computer_13")
async def computer13_mouse_move(
    worker_id: Annotated[str, Field(description="Worker id/session key (auto-injected by HttpMCPEnv).")],
    x: Annotated[int, Field(description="Target X coordinate in pixels (absolute).")],
    y: Annotated[int, Field(description="Target Y coordinate in pixels (absolute).")]
) -> list:
    """Move mouse to absolute coordinates (x, y)."""
    ctrl = _ctrl(worker_id)
    return await _execute_and_capture(worker_id, lambda: ctrl.execute_action({
        "action_type": "MOVE_TO",
        "parameters": {"x": x, "y": y}
    }))


@ToolRegistry.register_tool("desktop_action_computer_13")
async def computer13_mouse_click(
    worker_id: Annotated[str, Field(description="Worker id/session key (auto-injected by HttpMCPEnv).")],
    x: Annotated[int | None, Field(description="Optional absolute X coordinate. Omit to click at current cursor.")] = None,
    y: Annotated[int | None, Field(description="Optional absolute Y coordinate. Omit to click at current cursor.")] = None,
    button: Annotated[Literal["left", "right", "middle"], Field(description="Mouse button to click.")] = "left",
    num_clicks: Annotated[int, Field(description="Number of clicks to perform.", ge=1, le=3)] = 1
) -> list:
    """Click mouse at optional coordinates; supports left/right/middle and multi-click (1-3)."""
    ctrl = _ctrl(worker_id)
    params: dict[str, Any] = {"button": button}
    if num_clicks and num_clicks != 1:
        params["num_clicks"] = num_clicks
    if x is not None and y is not None:
        params["x"] = x
        params["y"] = y
    return await _execute_and_capture(worker_id, lambda: ctrl.execute_action({
        "action_type": "CLICK",
        "parameters": params
    }))


@ToolRegistry.register_tool("desktop_action_computer_13")
async def computer13_mouse_right_click(
    worker_id: Annotated[str, Field(description="Worker id/session key (auto-injected by HttpMCPEnv).")],
    x: Annotated[int | None, Field(description="Optional absolute X coordinate.")] = None,
    y: Annotated[int | None, Field(description="Optional absolute Y coordinate.")] = None
) -> list:
    """Right-click optionally at (x, y); defaults to current cursor if coordinates omitted."""
    ctrl = _ctrl(worker_id)
    params: dict[str, Any] = {}
    if x is not None and y is not None:
        params["x"] = x
        params["y"] = y
    return await _execute_and_capture(worker_id, lambda: ctrl.execute_action({
        "action_type": "RIGHT_CLICK",
        "parameters": params
    }))


@ToolRegistry.register_tool("desktop_action_computer_13")
async def computer13_mouse_double_click(
    worker_id: Annotated[str, Field(description="Worker id/session key (auto-injected by HttpMCPEnv).")],
    x: Annotated[int | None, Field(description="Optional absolute X coordinate.")] = None,
    y: Annotated[int | None, Field(description="Optional absolute Y coordinate.")] = None
) -> list:
    """Double-click optionally at (x, y); defaults to cursor position if not provided."""
    ctrl = _ctrl(worker_id)
    params: dict[str, Any] = {}
    if x is not None and y is not None:
        params["x"] = x
        params["y"] = y
    return await _execute_and_capture(worker_id, lambda: ctrl.execute_action({
        "action_type": "DOUBLE_CLICK",
        "parameters": params
    }))


@ToolRegistry.register_tool("desktop_action_computer_13")
async def computer13_mouse_down(
    worker_id: Annotated[str, Field(description="Worker id/session key (auto-injected by HttpMCPEnv).")],
    button: Annotated[Literal["left", "right", "middle"], Field(description="Mouse button to hold.")] = "left"
) -> list:
    """Press and hold a mouse button (default: left)."""
    ctrl = _ctrl(worker_id)
    return await _execute_and_capture(worker_id, lambda: ctrl.execute_action({
        "action_type": "MOUSE_DOWN",
        "parameters": {"button": button}
    }))


@ToolRegistry.register_tool("desktop_action_computer_13")
async def computer13_mouse_up(
    worker_id: Annotated[str, Field(description="Worker id/session key (auto-injected by HttpMCPEnv).")],
    button: Annotated[Literal["left", "right", "middle"], Field(description="Mouse button to release.")] = "left"
) -> list:
    """Release a mouse button (default: left)."""
    ctrl = _ctrl(worker_id)
    return await _execute_and_capture(worker_id, lambda: ctrl.execute_action({
        "action_type": "MOUSE_UP",
        "parameters": {"button": button}
    }))


@ToolRegistry.register_tool("desktop_action_computer_13")
async def computer13_drag_to(
    worker_id: Annotated[str, Field(description="Worker id/session key (auto-injected by HttpMCPEnv).")],
    x: Annotated[int, Field(description="Target X coordinate in pixels (absolute).")],
    y: Annotated[int, Field(description="Target Y coordinate in pixels (absolute).")]
) -> list:
    """Drag to absolute coordinates (x, y)."""
    ctrl = _ctrl(worker_id)
    return await _execute_and_capture(worker_id, lambda: ctrl.execute_action({
        "action_type": "DRAG_TO",
        "parameters": {"x": x, "y": y}
    }))


@ToolRegistry.register_tool("desktop_action_computer_13")
async def computer13_scroll(
    worker_id: Annotated[str, Field(description="Worker id/session key (auto-injected by HttpMCPEnv).")],
    dx: Annotated[int, Field(description="Horizontal scroll delta; positive=right, negative=left.")] = 0,
    dy: Annotated[int, Field(description="Vertical scroll delta; positive=up, negative=down.")] = 0
) -> list:
    """Scroll horizontally (dx) and/or vertically (dy)."""
    ctrl = _ctrl(worker_id)
    params: dict[str, int] = {}
    if dx:
        params["dx"] = dx
    if dy:
        params["dy"] = dy
    if not params:
        params = {"dy": -1}
    return await _execute_and_capture(worker_id, lambda: ctrl.execute_action({
        "action_type": "SCROLL",
        "parameters": params
    }))


@ToolRegistry.register_tool("desktop_action_computer_13")
async def computer13_type_text(
    worker_id: Annotated[str, Field(description="Worker id/session key (auto-injected by HttpMCPEnv).")],
    text: Annotated[str, Field(description="Text to type into the focused element.", min_length=1)]
) -> list:
    """Type a string into the focused element."""
    ctrl = _ctrl(worker_id)
    return await _execute_and_capture(worker_id, lambda: ctrl.execute_action({
        "action_type": "TYPING",
        "parameters": {"text": text}
    }))


@ToolRegistry.register_tool("desktop_action_computer_13")
async def computer13_key_press(
    worker_id: Annotated[str, Field(description="Worker id/session key (auto-injected by HttpMCPEnv).")],
    key: Annotated[str, Field(description="Single key name (lowercase, must be in KEYBOARD_KEYS).", min_length=1)]
) -> list:
    """Press a single key (e.g., 'enter')."""
    ctrl = _ctrl(worker_id)
    return await _execute_and_capture(worker_id, lambda: ctrl.execute_action({
        "action_type": "PRESS",
        "parameters": {"key": key}
    }))


@ToolRegistry.register_tool("desktop_action_computer_13")
async def computer13_key_down(
    worker_id: Annotated[str, Field(description="Worker id/session key (auto-injected by HttpMCPEnv).")],
    key: Annotated[str, Field(description="Key name to hold (lowercase, must be in KEYBOARD_KEYS).", min_length=1)]
) -> list:
    """Hold a key down."""
    ctrl = _ctrl(worker_id)
    return await _execute_and_capture(worker_id, lambda: ctrl.execute_action({
        "action_type": "KEY_DOWN",
        "parameters": {"key": key}
    }))


@ToolRegistry.register_tool("desktop_action_computer_13")
async def computer13_key_up(
    worker_id: Annotated[str, Field(description="Worker id/session key (auto-injected by HttpMCPEnv).")],
    key: Annotated[str, Field(description="Key name to release (lowercase, must be in KEYBOARD_KEYS).", min_length=1)]
) -> list:
    """Release a key."""
    ctrl = _ctrl(worker_id)
    return await _execute_and_capture(worker_id, lambda: ctrl.execute_action({
        "action_type": "KEY_UP",
        "parameters": {"key": key}
    }))


@ToolRegistry.register_tool("desktop_action_computer_13")
async def computer13_hotkey(
    worker_id: Annotated[str, Field(description="Worker id/session key (auto-injected by HttpMCPEnv).")],
    keys: Annotated[List[str], Field(description="Ordered list of keys for hotkey combo, e.g., ['ctrl','c'].", min_length=1)]
) -> list:
    """Press a combination of keys (e.g., ['ctrl','c'])."""
    ctrl = _ctrl(worker_id)
    return await _execute_and_capture(worker_id, lambda: ctrl.execute_action({
        "action_type": "HOTKEY",
        "parameters": {"keys": keys}
    }))


@ToolRegistry.register_tool("desktop_action_computer_13")
async def computer13_wait(
    worker_id: Annotated[str, Field(description="Worker id/session key (auto-injected by HttpMCPEnv).")],
    seconds: Annotated[float, Field(description="Seconds to pause before capturing observation.", gt=0, le=30.0)] = 1.0
) -> list:
    """Wait for N seconds and capture the current screen and a11y tree."""
    ctrl = _ctrl(worker_id)
    return await _execute_and_capture(worker_id, lambda: ctrl.execute_python_command(f"import time; time.sleep({seconds})"))
