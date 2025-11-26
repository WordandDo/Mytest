# src/mcp_server/osworld_server.py
import sys
import os
import base64
import json
import httpx
import asyncio
from typing import Optional, List
from dotenv import load_dotenv
import fastmcp
load_dotenv()
cwd = os.getcwd()
sys.path.append(cwd)
sys.path.append(os.path.join(cwd, "src"))

from mcp.server.fastmcp import FastMCP
from src.utils.desktop_env.controllers.python import PythonController

mcp = FastMCP("OSWorld Specialized Gateway")
RESOURCE_API_URL = os.environ.get("RESOURCE_API_URL", "http://localhost:8000")
ACTION_SPACE = os.environ.get("ACTION_SPACE", "computer_13")

print(f"🚀 Starting OSWorld MCP Server in [{ACTION_SPACE}] mode")

# [关键修改] 全局会话字典，Key 为 worker_id
GLOBAL_SESSIONS = {}

def _get_controller(worker_id: str) -> PythonController:
    session = GLOBAL_SESSIONS.get(worker_id)
    if not session or not session.get("controller"):
        raise RuntimeError(f"Session not found for worker: {worker_id}. Call 'setup_environment' first.")
    return session["controller"]

# --- 生命周期工具 ---

@mcp.tool()
async def setup_environment(config_name: str, task_id: str, worker_id: str) -> str:
    """初始化环境：申请资源并连接。必须提供 worker_id。"""
    async with httpx.AsyncClient() as client:
        try:
            # Resource API 的 allocate 是幂等的，可以安全重试
            resp = await client.post(f"{RESOURCE_API_URL}/allocate", json={"worker_id": worker_id}, timeout=120)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return json.dumps({"status": "error", "message": f"Alloc failed: {e}"})

    env_id = data.get("id")
    ip = data.get("ip")
    port = data.get("port", 5000)

    try:
        # 初始化控制器
        controller = PythonController(vm_ip=ip, server_port=port)
        
        # 存入全局会话
        GLOBAL_SESSIONS[worker_id] = {
            "controller": controller,
            "env_id": env_id,
            "task_id": task_id
        }
        
        # 获取初始状态
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
async def teardown_environment(worker_id: str) -> str:
    """释放资源"""
    session = GLOBAL_SESSIONS.get(worker_id)
    if session:
        env_id = session.get("env_id")
        async with httpx.AsyncClient() as client:
            try:
                await client.post(f"{RESOURCE_API_URL}/release", 
                                json={"resource_id": env_id, "worker_id": worker_id}, timeout=10)
            except:
                pass
        GLOBAL_SESSIONS.pop(worker_id, None)
    return "Released"

@mcp.tool()
async def get_observation(worker_id: str) -> str:
    """获取当前屏幕状态"""
    ctrl = _get_controller(worker_id)
    screenshot = ctrl.get_screenshot()
    shot_b64 = base64.b64encode(screenshot).decode('utf-8') if screenshot else ""
    return json.dumps({
        "screenshot": shot_b64,
        "accessibility_tree": ctrl.get_accessibility_tree()
    })

@mcp.tool()
async def evaluate_task(worker_id: str) -> str:
    return "0.0"

# --- 动作工具 (Computer 13 Mode) ---
# 所有工具均增加了 worker_id 参数

if ACTION_SPACE == "computer_13":

    @mcp.tool()
    async def desktop_mouse_move(worker_id: str, x: Optional[int] = None, y: Optional[int] = None) -> str:
        ctrl = _get_controller(worker_id)
        params = {}
        if x is not None and y is not None:
            params = {"x": x, "y": y}
        ctrl.execute_action({"action_type": "MOVE_TO", "parameters": params})
        return json.dumps({"status": "success", "action": "MOVE_TO"})

    @mcp.tool()
    async def desktop_mouse_click(worker_id: str, x: Optional[int] = None, y: Optional[int] = None, button: str = "left", num_clicks: int = 1) -> str:
        ctrl = _get_controller(worker_id)
        params = {"button": button, "num_clicks": num_clicks}
        if x is not None and y is not None:
            params.update({"x": x, "y": y})
        ctrl.execute_action({"action_type": "CLICK", "parameters": params})
        return json.dumps({"status": "success", "action": "CLICK"})

    @mcp.tool()
    async def desktop_mouse_button(worker_id: str, action: str, button: str = "left") -> str:
        ctrl = _get_controller(worker_id)
        act_type = "MOUSE_DOWN" if action.lower() == "down" else "MOUSE_UP"
        ctrl.execute_action({"action_type": act_type, "parameters": {"button": button}})
        return json.dumps({"status": "success", "action": act_type})

    @mcp.tool()
    async def desktop_mouse_right_click(worker_id: str, x: Optional[int] = None, y: Optional[int] = None) -> str:
        ctrl = _get_controller(worker_id)
        params = {}
        if x is not None and y is not None:
            params = {"x": x, "y": y}
        ctrl.execute_action({"action_type": "RIGHT_CLICK", "parameters": params})
        return json.dumps({"status": "success", "action": "RIGHT_CLICK"})

    @mcp.tool()
    async def desktop_mouse_double_click(worker_id: str, x: Optional[int] = None, y: Optional[int] = None) -> str:
        ctrl = _get_controller(worker_id)
        params = {}
        if x is not None and y is not None:
            params = {"x": x, "y": y}
        ctrl.execute_action({"action_type": "DOUBLE_CLICK", "parameters": params})
        return json.dumps({"status": "success", "action": "DOUBLE_CLICK"})

    @mcp.tool()
    async def desktop_mouse_drag(worker_id: str, x: int, y: int) -> str:
        ctrl = _get_controller(worker_id)
        ctrl.execute_action({"action_type": "DRAG_TO", "parameters": {"x": x, "y": y}})
        return json.dumps({"status": "success", "action": "DRAG_TO"})

    @mcp.tool()
    async def desktop_scroll(worker_id: str, dx: Optional[int] = None, dy: Optional[int] = None) -> str:
        ctrl = _get_controller(worker_id)
        params = {}
        if dx is not None: params["dx"] = dx
        if dy is not None: params["dy"] = dy
        ctrl.execute_action({"action_type": "SCROLL", "parameters": params})
        return json.dumps({"status": "success", "action": "SCROLL"})

    @mcp.tool()
    async def desktop_type(worker_id: str, text: str) -> str:
        ctrl = _get_controller(worker_id)
        ctrl.execute_action({"action_type": "TYPING", "parameters": {"text": text}})
        return json.dumps({"status": "success", "action": "TYPING"})

    @mcp.tool()
    async def desktop_key_press(worker_id: str, key: str) -> str:
        ctrl = _get_controller(worker_id)
        ctrl.execute_action({"action_type": "PRESS", "parameters": {"key": key}})
        return json.dumps({"status": "success", "action": "PRESS"})

    @mcp.tool()
    async def desktop_key_hold(worker_id: str, key: str, action: str) -> str:
        ctrl = _get_controller(worker_id)
        act_type = "KEY_DOWN" if action.lower() == "down" else "KEY_UP"
        ctrl.execute_action({"action_type": act_type, "parameters": {"key": key}})
        return json.dumps({"status": "success", "action": act_type})

    @mcp.tool()
    async def desktop_hotkey(worker_id: str, keys: List[str]) -> str:
        ctrl = _get_controller(worker_id)
        ctrl.execute_action({"action_type": "HOTKEY", "parameters": {"keys": keys}})
        return json.dumps({"status": "success", "action": "HOTKEY"})

    @mcp.tool()
    async def desktop_control(worker_id: str, action: str) -> str:
        ctrl = _get_controller(worker_id)
        act_str = action.upper()
        ctrl.execute_action(act_str)
        return json.dumps({"status": "success", "action": act_str})

# --- PyAutoGUI Mode ---
elif ACTION_SPACE == "pyautogui":
    @mcp.tool()
    async def desktop_execute_python_script(worker_id: str, script: str) -> str:
        ctrl = _get_controller(worker_id)
        try:
            result = ctrl.execute_python_command(script)
            return json.dumps({"status": "success", "output": result})
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def desktop_control(worker_id: str, action: str) -> str:
        ctrl = _get_controller(worker_id)
        act_str = action.upper()
        ctrl.execute_action(act_str)
        return json.dumps({"status": "success", "action": act_str})


if __name__ == "__main__":
    import uvicorn
    
    mcp.settings.debug = True
    print("🚀 Starting MCP Server on port 8080 (SSE Mode)...")
    mcp.run(transport='sse', host="0.0.0.0", port=8080)