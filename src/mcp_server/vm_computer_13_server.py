# src/mcp_server/vm_computer_13_server.py
import sys
import os
import base64
import json
import httpx
import asyncio
import logging
import time # 新增导入
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
from src.utils.desktop_env.controllers.setup import execute_setup_steps # 确保导入 execute_setup_steps

# 导入注册表
from mcp_server.core.registry import ToolRegistry

# 设置日志
logger = logging.getLogger("VMComputer13Server")

# 设置服务器名称为资源专属名称
mcp = FastMCP("VM Computer 13 Gateway")
RESOURCE_API_URL = os.environ.get("RESOURCE_API_URL", "http://localhost:8000")

print(f"🚀 Starting VM Computer 13 MCP Server (Registry Mode)")

# 全局会话字典，Key 为 worker_id
GLOBAL_SESSIONS = {}

# --- 通用功能提取 (与 os_pyautogui_server 保持一致) ---

async def vm_computer_13_initialization(worker_id: str, config_content = None) -> bool:
    """
    VM Computer 13 资源初始化函数，用于解析Benchmark特有的数据结构并执行初始化操作
    """
    # 防御性编程：无配置即成功
    if not config_content:
        logger.info(f"[{worker_id}] VM initialization skipped: no config_content provided")
        return True

    try:
        logger.info(f"[{worker_id}] VM initialization started. config_content type: {type(config_content)}")

        session = GLOBAL_SESSIONS.get(worker_id)
        if not session or not session.get("controller"):
            # Session未找到，尝试调用 setup_vm_session 工具进行初始化
            try:
                # 如果 config_content 是 dict，转为 JSON 字符串
                init_script = json.dumps(config_content) if isinstance(config_content, dict) else config_content
                logger.info(f"[{worker_id}] Session not found, calling setup_vm_session")
                # 注意：此处自动初始化仍调用 setup_vm_session，它会硬编码资源类型
                result_json = await setup_computer_13_session(
                    config_name="auto_init",
                    task_id="unknown",
                    worker_id=worker_id,
                    init_script=init_script
                )
                result = json.loads(result_json)
                success = result.get("status") == "success"
                logger.info(f"[{worker_id}] setup_vm_session result: {success}")
                return success
            except Exception as e:
                logger.error(f"[{worker_id}] Auto setup_vm_session failed: {e}", exc_info=True)
                return False

        # 如果 Session 存在，则手动执行配置逻辑
        controller = session["controller"]
        logger.info(f"[{worker_id}] Found existing session with controller")

        # 统一处理 config_content 可能是 dict 或 str 的情况
        if isinstance(config_content, dict):
            task_spec = config_content
        elif isinstance(config_content, str) and config_content.strip().startswith("{"):
            try:
                task_spec = json.loads(config_content)
            except json.JSONDecodeError as e:
                logger.error(f"[{worker_id}] Invalid JSON in init_script: {e}")
                raise RuntimeError(f"Invalid JSON in init_script: {e}")
        else:
            # Case C: 传入的是纯 Python 脚本字符串
            logger.info(f"[{worker_id}] Processing config_content as Python script")
            controller.execute_python_command(config_content)
            logger.info(f"[{worker_id}] VM initialization completed (script executed)")
            return True

        # 处理任务规范 (来自 Case A 或 Case B)
        setup_steps = task_spec.get("config", [])
        evaluator = task_spec.get("evaluator", {})
        logger.info(f"[{worker_id}] Task spec extracted: {len(setup_steps)} setup steps, evaluator present: {bool(evaluator)}")

        # 执行 config 中的每一步 (download, execute 等)
        if setup_steps:
            # 确保这里能正确导入 execute_setup_steps
            from src.utils.desktop_env.controllers.setup import execute_setup_steps
            execute_setup_steps(controller, setup_steps)
            logger.info(f"[{worker_id}] Setup steps completed")

        # 将 evaluator 缓存到 GLOBAL_SESSIONS 中供后续 evaluate_task 使用
        GLOBAL_SESSIONS[worker_id]["evaluator"] = evaluator
        logger.info(f"[{worker_id}] VM initialization completed successfully")

        return True
    except Exception as e:
        logger.error(f"[{worker_id}] VM initialization failed: {e}", exc_info=True)
        return False

def _get_controller(worker_id: str) -> PythonController:
    session = GLOBAL_SESSIONS.get(worker_id)
    if not session or not session.get("controller"):
        raise RuntimeError(f"Session not found for worker: {worker_id}. Call 'setup_computer_13_session' first.")
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

# --- 生命周期工具 (Group: computer_lifecycle) ---

@ToolRegistry.register_tool("computer_lifecycle", hidden=True)
async def setup_computer_13_session(config_name: str, task_id: str, worker_id: str, init_script: str = "") -> str:
    """
    [System Tool] Initialize VM Computer 13 session.
    Allocates VM resources and initializes the controller.
    
    Args:
        config_name: Configuration name.
        task_id: Task ID
        worker_id: Worker ID
        init_script: Initialization script content
    """
    
    # [关键修改] 硬编码目标资源类型为 vm_computer_13
    target_resource_type = "vm_computer_13"
    
    # 设置长超时
    req_timeout = 600.0 

    async with httpx.AsyncClient() as client:
        try:
            # 直接发起申请
            resp = await client.post(
                f"{RESOURCE_API_URL}/allocate",
                json={
                    "worker_id": worker_id, 
                    "type": target_resource_type, # 使用硬编码的资源类型
                    "timeout": req_timeout        
                },
                timeout=req_timeout + 5 
            )
            resp.raise_for_status()
            data = resp.json()
            
        except httpx.TimeoutException:
            return json.dumps({
                "status": "error", 
                "message": f"System busy: Could not acquire '{target_resource_type}' within {req_timeout}s. Resource queue timeout."
            })
        except httpx.HTTPStatusError as e:
            error_msg = f"Allocation failed: {e.response.text}"
            return json.dumps({"status": "error", "message": error_msg})
        except Exception as e:
            return json.dumps({"status": "error", "message": f"Network/Unknown error: {str(e)}"})

    env_id = data.get("id")
    ip = data.get("ip")
    port = data.get("port", 5000)

    try:
        # 初始化控制器
        controller = PythonController(vm_ip=ip, server_port=port)
        # time.sleep(3) # 根据需要保留或移除
        
        # 存入全局会话
        GLOBAL_SESSIONS[worker_id] = {
            "controller": controller,
            "env_id": env_id,
            "task_id": task_id
        }
        
        # 处理初始化脚本 (与通用逻辑相同)
        if init_script:
            if init_script.strip().startswith("{"):
                try:
                    task_spec = json.loads(init_script)
                    setup_steps = task_spec.get("config", [])
                    evaluator = task_spec.get("evaluator", {})
                    
                    if setup_steps:
                        from src.utils.desktop_env.controllers.setup import execute_setup_steps
                        execute_setup_steps(controller, setup_steps)
                    
                    GLOBAL_SESSIONS[worker_id]["evaluator"] = evaluator
                    
                except json.JSONDecodeError as e:
                    return json.dumps({"status": "error", "message": f"Invalid JSON in init_script: {e}"})
            else:
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

@ToolRegistry.register_tool("computer_lifecycle", hidden=True)
async def teardown_computer_13_environment(worker_id: str) -> str:
    """
    [System Tool] Teardown computer 13 environment.
    Releases resources associated with the session.
    """
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

@ToolRegistry.register_tool("computer_lifecycle", hidden=True)
async def evaluate_computer_13_task(worker_id: str) -> str:
    """
    [System Tool] Evaluate computer 13 task result.
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

@ToolRegistry.register_tool("desktop_observation", hidden=True)
async def start_computer_13_recording(worker_id: str) -> str:
    """
    [System Tool] Start screen recording for Computer 13.
    """
    try:
        ctrl = _get_controller(worker_id)
        ctrl.start_recording()
        return "Recording started"
    except Exception as e:
        return f"Failed to start recording: {str(e)}"

@ToolRegistry.register_tool("desktop_observation", hidden=True)
async def stop_computer_13_recording(worker_id: str, save_path: str) -> str:
    """
    [System Tool] Stop recording and save file for Computer 13.
    """
    try:
        ctrl = _get_controller(worker_id)
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        ctrl.end_recording(save_path)
        return f"Recording saved to {save_path}"
    except Exception as e:
        return f"Failed to stop recording: {str(e)}"

# =============================================================================
# Computer 13 专属动作 (Group: desktop_action_computer13)
# =============================================================================

@ToolRegistry.register_tool("desktop_action_computer13")
async def desktop_mouse_move(worker_id: str, x: Optional[int] = None, y: Optional[int] = None) -> list:
    """
    Move the mouse cursor to the specified coordinates.
    """
    ctrl = _get_controller(worker_id)
    params = {}
    if x is not None and y is not None:
        params = {"x": x, "y": y}
    
    return await _execute_and_capture(worker_id, lambda: 
        ctrl.execute_action({"action_type": "MOVE_TO", "parameters": params})
    )

@ToolRegistry.register_tool("desktop_action_computer13")
async def desktop_mouse_click(worker_id: str, x: Optional[int] = None, y: Optional[int] = None, button: str = "left", num_clicks: int = 1) -> list:
    """
    Click the mouse at the specified coordinates.
    """
    ctrl = _get_controller(worker_id)
    params = {"button": button, "num_clicks": num_clicks}
    if x is not None and y is not None:
        params.update({"x": x, "y": y})
    
    return await _execute_and_capture(worker_id, lambda: 
        ctrl.execute_action({"action_type": "CLICK", "parameters": params})
    )

@ToolRegistry.register_tool("desktop_action_computer13")
async def desktop_mouse_right_click(worker_id: str, x: Optional[int] = None, y: Optional[int] = None) -> list:
    """
    Right-click the mouse at the specified coordinates.
    """
    ctrl = _get_controller(worker_id)
    params = {}
    if x is not None and y is not None:
        params = {"x": x, "y": y}
    
    return await _execute_and_capture(worker_id, lambda: 
        ctrl.execute_action({"action_type": "RIGHT_CLICK", "parameters": params})
    )

@ToolRegistry.register_tool("desktop_action_computer13")
async def desktop_mouse_double_click(worker_id: str, x: Optional[int] = None, y: Optional[int] = None) -> list:
    """
    Double-click the mouse at the specified coordinates.
    """
    ctrl = _get_controller(worker_id)
    params = {}
    if x is not None and y is not None:
        params = {"x": x, "y": y}
    
    return await _execute_and_capture(worker_id, lambda: 
        ctrl.execute_action({"action_type": "DOUBLE_CLICK", "parameters": params})
    )

@ToolRegistry.register_tool("desktop_action_computer13")
async def desktop_mouse_drag(worker_id: str, x: int, y: int) -> list:
    """
    Drag the mouse to the specified coordinates.
    """
    ctrl = _get_controller(worker_id)
    return await _execute_and_capture(worker_id, lambda: 
        ctrl.execute_action({"action_type": "DRAG_TO", "parameters": {"x": x, "y": y}})
    )

@ToolRegistry.register_tool("desktop_action_computer13")
async def desktop_scroll(worker_id: str, dx: Optional[int] = None, dy: Optional[int] = None) -> list:
    """
    Scroll the mouse wheel by the specified amount.
    """
    ctrl = _get_controller(worker_id)
    params = {}
    if dx is not None: params["dx"] = dx
    if dy is not None: params["dy"] = dy
    
    return await _execute_and_capture(worker_id, lambda: 
        ctrl.execute_action({"action_type": "SCROLL", "parameters": params})
    )

@ToolRegistry.register_tool("desktop_action_computer13")
async def desktop_type(worker_id: str, text: str) -> list:
    """
    Type the specified text.
    """
    ctrl = _get_controller(worker_id)
    return await _execute_and_capture(worker_id, lambda: 
        ctrl.execute_action({"action_type": "TYPING", "parameters": {"text": text}})
    )

@ToolRegistry.register_tool("desktop_action_computer13")
async def desktop_key_press(worker_id: str, key: str) -> list:
    """
    Press the specified key.
    """
    ctrl = _get_controller(worker_id)
    return await _execute_and_capture(worker_id, lambda: 
        ctrl.execute_action({"action_type": "PRESS", "parameters": {"key": key}})
    )

@ToolRegistry.register_tool("desktop_action_computer13")
async def desktop_key_hold(worker_id: str, key: str, action: str) -> list:
    """
    Hold or release the specified key.
    """
    ctrl = _get_controller(worker_id)
    act_type = "KEY_DOWN" if action.lower() == "down" else "KEY_UP"
    return await _execute_and_capture(worker_id, lambda: 
        ctrl.execute_action({"action_type": act_type, "parameters": {"key": key}})
    )

@ToolRegistry.register_tool("desktop_action_computer13")
async def desktop_hotkey(worker_id: str, keys: List[str]) -> list:
    """
    Press a combination of keys simultaneously.
    """
    ctrl = _get_controller(worker_id)
    return await _execute_and_capture(worker_id, lambda: 
        ctrl.execute_action({"action_type": "HOTKEY", "parameters": {"keys": keys}})
    )

# 共享动作 (只注册到 desktop_action_computer13)
@ToolRegistry.register_tool("desktop_action_computer13")
async def desktop_mouse_button(worker_id: str, action: str, button: str = "left") -> list:
    """
    Press down or release the mouse button.
    """
    ctrl = _get_controller(worker_id)
    act_type = "MOUSE_DOWN" if action.lower() == "down" else "MOUSE_UP"
    return await _execute_and_capture(worker_id, lambda: 
        ctrl.execute_action({"action_type": act_type, "parameters": {"button": button}})
    )

@ToolRegistry.register_tool("desktop_action_computer13")
async def desktop_control(worker_id: str, action: str) -> list:
    """
    Execute a control action.
    """
    ctrl = _get_controller(worker_id)
    act_str = action.upper()
    return await _execute_and_capture(worker_id, lambda: 
        ctrl.execute_action(act_str)
    )