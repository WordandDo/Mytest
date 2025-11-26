# src/mcp_server/osworld_server.py
import sys
import os
import base64
import json
import httpx
import asyncio
from typing import Optional, List
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 1. 路径修正：确保能找到 src.utils
cwd = os.getcwd()
sys.path.append(cwd)
sys.path.append(os.path.join(cwd, "src"))

from mcp.server.fastmcp import FastMCP
from src.utils.desktop_env.controllers.python import PythonController

# 初始化 MCP Server
mcp = FastMCP("OSWorld Specialized Gateway")

# 资源 API 地址
RESOURCE_API_URL = "http://localhost:8000"

# 获取当前模式 (computer_13 或 pyautogui)
ACTION_SPACE = os.environ.get("ACTION_SPACE", "computer_13")

print(f"🚀 Starting OSWorld MCP Server in [{ACTION_SPACE}] mode")

# 会话状态
current_session = {
    "controller": None,
    "env_id": None,
    "task_id": None
}

def _get_controller() -> PythonController:
    ctrl = current_session.get("controller")
    if not ctrl:
        raise RuntimeError("Environment not initialized. Call 'setup_environment' first.")
    return ctrl

# ----------------------------------------------------------------
# 1. 生命周期管理 (通用)
# ----------------------------------------------------------------

@mcp.tool()
async def setup_environment(config_name: str, task_id: str) -> str:
    """
    初始化环境：申请资源并连接。
    返回包含初始截图和 Accessibility Tree 的 JSON。
    """
    async with httpx.AsyncClient() as client:
        try:
            # 申请资源
            resp = await client.post(f"{RESOURCE_API_URL}/allocate", json={"worker_id": task_id}, timeout=120)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return json.dumps({"status": "error", "message": f"Alloc failed: {e}"})

    env_id = data.get("id")
    ip = data.get("ip")
    port = data.get("port", 5000)

    if not ip:
        return json.dumps({"status": "error", "message": "No IP returned"})

    try:
        # 连接底层控制器
        controller = PythonController(vm_ip=ip, server_port=port)
        current_session.update({"controller": controller, "env_id": env_id, "task_id": task_id})
        
        # 获取初始状态 (带 Base64 修复)
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

@mcp.tool()
async def teardown_environment() -> str:
    """释放资源并清理环境"""
    env_id = current_session.get("env_id")
    task_id = current_session.get("task_id")
    if env_id:
        async with httpx.AsyncClient() as client:
            try:
                await client.post(f"{RESOURCE_API_URL}/release", 
                                json={"resource_id": env_id, "worker_id": task_id}, timeout=10)
            except:
                pass
    current_session["controller"] = None
    return "Released"

@mcp.tool()
async def get_observation() -> str:
    """获取当前屏幕状态（截图和 Accessibility Tree）。"""
    ctrl = _get_controller()
    
    screenshot = ctrl.get_screenshot()
    # Base64 编码修复
    shot_b64 = base64.b64encode(screenshot).decode('utf-8') if screenshot else ""
    
    return json.dumps({
        "screenshot": shot_b64,
        "accessibility_tree": ctrl.get_accessibility_tree()
    })

@mcp.tool()
async def evaluate_task() -> str:
    """获取当前任务评分 (如有配置)"""
    return "0.0"

# =================================================================
# 模式 A: Computer 13 (完整原子工具组 - 12个)
# 对应 src/tools/osworld_tools.py 中的定义
# =================================================================
if ACTION_SPACE == "computer_13":

    @mcp.tool()
    async def desktop_mouse_move(x: Optional[int] = None, y: Optional[int] = None) -> str:
        """
        Move the cursor to the specified position. 
        Both x and y must be provided together, or omit both for default position.
        """
        ctrl = _get_controller()
        params = {}
        if x is not None and y is not None:
            params = {"x": x, "y": y}
        
        action = {"action_type": "MOVE_TO", "parameters": params}
        ctrl.execute_action(action)
        return json.dumps({"status": "success", "action": "MOVE_TO"})

    @mcp.tool()
    async def desktop_mouse_click(
        x: Optional[int] = None, 
        y: Optional[int] = None, 
        button: str = "left", 
        num_clicks: int = 1
    ) -> str:
        """
        Click mouse button. 
        If x/y provided, moves there first. Default: left button, 1 click.
        """
        ctrl = _get_controller()
        params = {"button": button, "num_clicks": num_clicks}
        if x is not None and y is not None:
            params.update({"x": x, "y": y})

        action = {"action_type": "CLICK", "parameters": params}
        ctrl.execute_action(action)
        return json.dumps({"status": "success", "action": "CLICK"})

    @mcp.tool()
    async def desktop_mouse_button(action: str, button: str = "left") -> str:
        """
        Press or release mouse button.
        Args:
            action: 'down' (press) or 'up' (release). REQUIRED.
            button: 'left', 'right', 'middle'. Default 'left'.
        """
        ctrl = _get_controller()
        # PythonController expecting MOUSE_DOWN or MOUSE_UP based on action_type string
        # But in python.py logic: action_type == "MOUSE_DOWN" -> execute_python_command("pyautogui.mouseDown()")
        
        act_type = "MOUSE_DOWN" if action.lower() == "down" else "MOUSE_UP"
        
        action_payload = {
            "action_type": act_type,
            "parameters": {"button": button}
        }
        ctrl.execute_action(action_payload)
        return json.dumps({"status": "success", "action": act_type})

    @mcp.tool()
    async def desktop_mouse_right_click(x: Optional[int] = None, y: Optional[int] = None) -> str:
        """Right-click at position (or current position if x,y omitted)."""
        ctrl = _get_controller()
        params = {}
        if x is not None and y is not None:
            params = {"x": x, "y": y}

        action = {"action_type": "RIGHT_CLICK", "parameters": params}
        ctrl.execute_action(action)
        return json.dumps({"status": "success", "action": "RIGHT_CLICK"})

    @mcp.tool()
    async def desktop_mouse_double_click(x: Optional[int] = None, y: Optional[int] = None) -> str:
        """Double-click at position (or current position if x,y omitted)."""
        ctrl = _get_controller()
        params = {}
        if x is not None and y is not None:
            params = {"x": x, "y": y}

        action = {"action_type": "DOUBLE_CLICK", "parameters": params}
        ctrl.execute_action(action)
        return json.dumps({"status": "success", "action": "DOUBLE_CLICK"})

    @mcp.tool()
    async def desktop_mouse_drag(x: int, y: int) -> str:
        """Drag mouse to position (x, y) with left button pressed."""
        ctrl = _get_controller()
        action = {
            "action_type": "DRAG_TO",
            "parameters": {"x": x, "y": y}
        }
        ctrl.execute_action(action)
        return json.dumps({"status": "success", "action": "DRAG_TO"})

    @mcp.tool()
    async def desktop_scroll(dx: Optional[int] = None, dy: Optional[int] = None) -> str:
        """
        Scroll the mouse wheel.
        At least one of dx (horizontal) or dy (vertical) is required.
        """
        ctrl = _get_controller()
        params = {}
        if dx is not None: params["dx"] = dx
        if dy is not None: params["dy"] = dy
        
        if not params:
            return json.dumps({"status": "error", "message": "Either dx or dy must be provided"})

        action = {"action_type": "SCROLL", "parameters": params}
        ctrl.execute_action(action)
        return json.dumps({"status": "success", "action": "SCROLL"})

    @mcp.tool()
    async def desktop_type(text: str) -> str:
        """Type text string."""
        ctrl = _get_controller()
        action = {
            "action_type": "TYPING",
            "parameters": {"text": text}
        }
        ctrl.execute_action(action)
        return json.dumps({"status": "success", "action": "TYPING"})

    @mcp.tool()
    async def desktop_key_press(key: str) -> str:
        """Press and release a key (e.g., 'enter', 'space', 'backspace')."""
        ctrl = _get_controller()
        action = {
            "action_type": "PRESS",
            "parameters": {"key": key}
        }
        ctrl.execute_action(action)
        return json.dumps({"status": "success", "action": "PRESS"})

    @mcp.tool()
    async def desktop_key_hold(key: str, action: str) -> str:
        """
        Press or release a key.
        Args:
            key: The key name.
            action: 'down' (press) or 'up' (release).
        """
        ctrl = _get_controller()
        act_type = "KEY_DOWN" if action.lower() == "down" else "KEY_UP"
        
        action_payload = {
            "action_type": act_type,
            "parameters": {"key": key}
        }
        ctrl.execute_action(action_payload)
        return json.dumps({"status": "success", "action": act_type})

    @mcp.tool()
    async def desktop_hotkey(keys: List[str]) -> str:
        """Press key combination (e.g. ['ctrl', 'c'])."""
        ctrl = _get_controller()
        action = {
            "action_type": "HOTKEY",
            "parameters": {"keys": keys}
        }
        ctrl.execute_action(action)
        return json.dumps({"status": "success", "action": "HOTKEY"})

    @mcp.tool()
    async def desktop_control(action: str) -> str:
        """
        Send control signal.
        Args: action: 'wait', 'done', or 'fail'.
        """
        # Control actions don't execute on the VM controller generally,
        # but serve as a signal to the agent loop or system.
        # python.py's execute_action returns immediately for these types.
        ctrl = _get_controller()
        
        # python.py expects action to be a string like 'WAIT', not a dict with action_type='WAIT'
        # See python.py: if action in ['WAIT', 'FAIL', 'DONE']: return
        
        act_str = action.upper()
        if act_str not in ['WAIT', 'DONE', 'FAIL']:
            return json.dumps({"status": "error", "message": "Invalid action"})
            
        # Since execute_action(str) is valid in python.py for these special cases
        ctrl.execute_action(act_str)
        
        return json.dumps({"status": "success", "action": act_str})


# =================================================================
# 模式 B: PyAutoGUI (脚本执行模式)
# =================================================================
elif ACTION_SPACE == "pyautogui":

    @mcp.tool()
    async def desktop_execute_python_script(script: str) -> str:
        """
        Executes Python code using `pyautogui`.
        Specify coordinates directly. DO NOT use image recognition.
        """
        ctrl = _get_controller()
        # 直接调用 execute_python_command 接口
        try:
            result = ctrl.execute_python_command(script)
            return json.dumps({"status": "success", "output": result})
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def desktop_control(action: str) -> str:
        """
        Send control signal: 'wait', 'done', or 'fail'.
        """
        ctrl = _get_controller()
        act_str = action.upper()
        if act_str not in ['WAIT', 'DONE', 'FAIL']:
            return json.dumps({"status": "error", "message": "Invalid action"})
        
        ctrl.execute_action(act_str)
        return json.dumps({"status": "success", "action": act_str})

if __name__ == "__main__":
    mcp.run()