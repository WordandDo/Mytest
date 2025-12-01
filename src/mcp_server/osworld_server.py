# src/mcp_server/osworld_server.py
import sys
import os
import base64
import json
import httpx
import asyncio
from typing import Optional, List
from dotenv import load_dotenv
load_dotenv()
cwd = os.getcwd()
sys.path.append(cwd)
sys.path.append(os.path.join(cwd, "src"))

from mcp.server.fastmcp import FastMCP
from src.utils.desktop_env.controllers.python import PythonController
from mcp_server.core.probe import wait_for_resource_availability

# [新增] 导入注册表
from mcp_server.core.registry import ToolRegistry

mcp = FastMCP("OSWorld Specialized Gateway")
RESOURCE_API_URL = os.environ.get("RESOURCE_API_URL", "http://localhost:8000")

print(f"🚀 Starting OSWorld MCP Server (Registry Mode)")

# [关键修改] 全局会话字典，Key 为 worker_id
GLOBAL_SESSIONS = {}

def _get_controller(worker_id: str) -> PythonController:
    session = GLOBAL_SESSIONS.get(worker_id)
    if not session or not session.get("controller"):
        raise RuntimeError(f"Session not found for worker: {worker_id}. Call 'setup_vm_session' first.")
    return session["controller"]

# --- 生命周期工具 (Group: computer_lifecycle) ---

@ToolRegistry.register_tool("computer_lifecycle")  # [新增注册]
async def setup_vm_session(config_name: str, task_id: str, worker_id: str, init_script: str = "") -> str:
    """初始化 VM 会话：申请 VM 资源并初始化控制器。
    (原名 setup_environment，已重命名以消除歧义)
    """
    
    # 1. 资源探活：在发起申请前，先确认有空闲资源
    # 避免盲目调用 /allocate 导致死锁或长时间 HTTP 挂起
    is_available = await wait_for_resource_availability(
        api_url=RESOURCE_API_URL,
        resource_type="vm",
        timeout=30  # 等待 30 秒，如果还没有释放则报错
    )
    
    if not is_available:
        return json.dumps({
            "status": "error", 
            "message": "System busy: No VM resources available. Please try again later."
        })

    # 2. 正式申请资源 (原有逻辑)
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
        
        # 处理初始化脚本
        if init_script:
            # 判断是否是JSON格式的任务规范
            if init_script.strip().startswith("{"):
                # Case A: 传入的是 OSWorld 任务规范 (JSON)
                try:
                    task_spec = json.loads(init_script)
                    setup_steps = task_spec.get("config", [])
                    evaluator = task_spec.get("evaluator", {})
                    
                    # 执行 config 中的每一步 (download, execute 等)
                    if setup_steps:
                        from src.utils.desktop_env.controllers.setup import execute_setup_steps
                        execute_setup_steps(controller, setup_steps)
                    
                    # 将 evaluator 缓存到 GLOBAL_SESSIONS 中供后续 evaluate_task 使用
                    GLOBAL_SESSIONS[worker_id]["evaluator"] = evaluator
                    
                except json.JSONDecodeError as e:
                    return json.dumps({"status": "error", "message": f"Invalid JSON in init_script: {e}"})
            else:
                # Case B: 传入的是纯 Python 脚本 (如 Math/Web 任务)
                try:
                    controller.execute_python_command(init_script)
                except Exception as e:
                    return json.dumps({"status": "error", "message": f"Failed to execute init_script: {e}"})
        
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

@ToolRegistry.register_tool("computer_lifecycle") # [新增注册]
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

@ToolRegistry.register_tool("computer_lifecycle") # [新增注册] 归类为生命周期或评估
async def evaluate_task(worker_id: str) -> str:
    """评估任务执行结果"""
    session = GLOBAL_SESSIONS.get(worker_id)
    
    # 检查session是否存在以及是否有evaluator配置
    if not session or not session.get("evaluator"):
        # 安全返回，避免Crash
        return "0.0"
    
    # 获取评估器配置
    evaluator_config = session.get("evaluator", {})
    if not evaluator_config or not isinstance(evaluator_config, dict):
        return "0.0"
    
    # TODO: 实现具体的评估逻辑
    # 这里应该根据evaluator_config中的配置执行相应的评估函数
    # 例如调用check_include_exclude等评估方法
    
    # 临时返回默认分数
    return "0.0"

# --- 观察工具 (Group: desktop_observation) ---

@ToolRegistry.register_tool("desktop_observation") # [新增注册]
async def get_observation(worker_id: str) -> str:
    """获取当前屏幕状态"""
    ctrl = _get_controller(worker_id)
    screenshot = ctrl.get_screenshot()
    shot_b64 = base64.b64encode(screenshot).decode('utf-8') if screenshot else ""
    return json.dumps({
        "screenshot": shot_b64,
        "accessibility_tree": ctrl.get_accessibility_tree()
    })

@ToolRegistry.register_tool("desktop_observation")
async def start_recording(worker_id: str) -> str:
    """[新增] 开始屏幕录制"""
    try:
        ctrl = _get_controller(worker_id)
        ctrl.start_recording()
        return "Recording started"
    except Exception as e:
        return f"Failed to start recording: {str(e)}"

@ToolRegistry.register_tool("desktop_observation")
async def stop_recording(worker_id: str, save_path: str) -> str:
    """[新增] 停止录制并保存文件
    注意：save_path 是 Gateway 服务器本地的文件路径
    """
    try:
        ctrl = _get_controller(worker_id)
        # 确保目录存在
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        ctrl.end_recording(save_path)
        return f"Recording saved to {save_path}"
    except Exception as e:
        return f"Failed to stop recording: {str(e)}"

# --- 动作工具：拆分为 Computer 13, PyAutoGUI 和 Shared ---

# 1. Computer 13 专属动作
@ToolRegistry.register_tool("desktop_action_computer13")
async def desktop_mouse_move(worker_id: str, x: Optional[int] = None, y: Optional[int] = None) -> str:
    ctrl = _get_controller(worker_id)
    params = {}
    if x is not None and y is not None:
        params = {"x": x, "y": y}
    ctrl.execute_action({"action_type": "MOVE_TO", "parameters": params})
    return json.dumps({"status": "success", "action": "MOVE_TO"})

@ToolRegistry.register_tool("desktop_action_computer13")
async def desktop_mouse_click(worker_id: str, x: Optional[int] = None, y: Optional[int] = None, button: str = "left", num_clicks: int = 1) -> str:
    ctrl = _get_controller(worker_id)
    params = {"button": button, "num_clicks": num_clicks}
    if x is not None and y is not None:
        params.update({"x": x, "y": y})
    ctrl.execute_action({"action_type": "CLICK", "parameters": params})
    return json.dumps({"status": "success", "action": "CLICK"})

@ToolRegistry.register_tool("desktop_action_computer13")
async def desktop_mouse_right_click(worker_id: str, x: Optional[int] = None, y: Optional[int] = None) -> str:
    ctrl = _get_controller(worker_id)
    params = {}
    if x is not None and y is not None:
        params = {"x": x, "y": y}
    ctrl.execute_action({"action_type": "RIGHT_CLICK", "parameters": params})
    return json.dumps({"status": "success", "action": "RIGHT_CLICK"})

@ToolRegistry.register_tool("desktop_action_computer13")
async def desktop_mouse_double_click(worker_id: str, x: Optional[int] = None, y: Optional[int] = None) -> str:
    ctrl = _get_controller(worker_id)
    params = {}
    if x is not None and y is not None:
        params = {"x": x, "y": y}
    ctrl.execute_action({"action_type": "DOUBLE_CLICK", "parameters": params})
    return json.dumps({"status": "success", "action": "DOUBLE_CLICK"})

@ToolRegistry.register_tool("desktop_action_computer13")
async def desktop_mouse_drag(worker_id: str, x: int, y: int) -> str:
    ctrl = _get_controller(worker_id)
    ctrl.execute_action({"action_type": "DRAG_TO", "parameters": {"x": x, "y": y}})
    return json.dumps({"status": "success", "action": "DRAG_TO"})

@ToolRegistry.register_tool("desktop_action_computer13")
async def desktop_scroll(worker_id: str, dx: Optional[int] = None, dy: Optional[int] = None) -> str:
    ctrl = _get_controller(worker_id)
    params = {}
    if dx is not None: params["dx"] = dx
    if dy is not None: params["dy"] = dy
    ctrl.execute_action({"action_type": "SCROLL", "parameters": params})
    return json.dumps({"status": "success", "action": "SCROLL"})

@ToolRegistry.register_tool("desktop_action_computer13")
async def desktop_type(worker_id: str, text: str) -> str:
    ctrl = _get_controller(worker_id)
    ctrl.execute_action({"action_type": "TYPING", "parameters": {"text": text}})
    return json.dumps({"status": "success", "action": "TYPING"})

@ToolRegistry.register_tool("desktop_action_computer13")
async def desktop_key_press(worker_id: str, key: str) -> str:
    ctrl = _get_controller(worker_id)
    ctrl.execute_action({"action_type": "PRESS", "parameters": {"key": key}})
    return json.dumps({"status": "success", "action": "PRESS"})

@ToolRegistry.register_tool("desktop_action_computer13")
async def desktop_key_hold(worker_id: str, key: str, action: str) -> str:
    ctrl = _get_controller(worker_id)
    act_type = "KEY_DOWN" if action.lower() == "down" else "KEY_UP"
    ctrl.execute_action({"action_type": act_type, "parameters": {"key": key}})
    return json.dumps({"status": "success", "action": "KEY_ACTION"})

@ToolRegistry.register_tool("desktop_action_computer13")
async def desktop_hotkey(worker_id: str, keys: List[str]) -> str:
    ctrl = _get_controller(worker_id)
    ctrl.execute_action({"action_type": "HOTKEY", "parameters": {"keys": keys}})
    return json.dumps({"status": "success", "action": "HOTKEY"})

# 2. PyAutoGUI 专属动作
@ToolRegistry.register_tool("desktop_action_pyautogui")
async def desktop_execute_python_script(worker_id: str, script: str) -> str:
    ctrl = _get_controller(worker_id)
    try:
        result = ctrl.execute_python_command(script)
        return json.dumps({"status": "success", "output": result})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})

# 3. 共享动作 (注册到两个组)
@ToolRegistry.register_tool("desktop_action_computer13")
@ToolRegistry.register_tool("desktop_action_pyautogui")
async def desktop_mouse_button(worker_id: str, action: str, button: str = "left") -> str:
    # 注意：computer_13 用 MOUSE_DOWN/UP，PyAutoGUI 模式下 Controller 也应该能处理
    ctrl = _get_controller(worker_id)
    act_type = "MOUSE_DOWN" if action.lower() == "down" else "MOUSE_UP"
    ctrl.execute_action({"action_type": act_type, "parameters": {"button": button}})
    return json.dumps({"status": "success", "action": act_type})

@ToolRegistry.register_tool("desktop_action_computer13")
@ToolRegistry.register_tool("desktop_action_pyautogui")
async def desktop_control(worker_id: str, action: str) -> str:
    ctrl = _get_controller(worker_id)
    act_str = action.upper()
    ctrl.execute_action(act_str)
    return json.dumps({"status": "success", "action": act_str})