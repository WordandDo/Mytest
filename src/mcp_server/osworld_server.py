# src/mcp_server/osworld_server.py
import sys
import os
import base64
import json
import httpx
import asyncio
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

# [新增] 导入注册表
from mcp_server.core.registry import ToolRegistry

mcp = FastMCP("OSWorld Specialized Gateway")
RESOURCE_API_URL = os.environ.get("RESOURCE_API_URL", "http://localhost:8000")

print(f"🚀 Starting OSWorld MCP Server (Registry Mode)")

# [关键修改] 全局会话字典，Key 为 worker_id
GLOBAL_SESSIONS = {}

# [新增] VM初始化函数
async def vm_initialization(worker_id: str, config_content: str = "") -> bool:
    """
    VM资源初始化函数，用于解析Benchmark特有的数据结构并执行初始化操作
    
    Args:
        worker_id: 工作进程ID
        config_content: 初始化配置内容，可能是JSON格式或纯脚本
        
    Returns:
        bool: 初始化是否成功
    """
    # 防御性编程：无配置即成功
    if not config_content:
        return True
    
    try:
        session = GLOBAL_SESSIONS.get(worker_id)
        if not session or not session.get("controller"):
            # Session未找到，尝试调用 setup_vm_session 工具进行初始化
            # 注意：setup_vm_session 需要 config_name 和 task_id，此处作为自动初始化使用默认占位符
            try:
                result_json = await setup_vm_session(
                    config_name="auto_init", 
                    task_id="unknown", 
                    worker_id=worker_id, 
                    init_script=config_content
                )
                result = json.loads(result_json)
                return result.get("status") == "success"
            except Exception as e:
                print(f"Auto setup_vm_session failed for {worker_id}: {e}")
                return False
        
        # 如果 Session 存在，则手动执行配置逻辑
        controller = session["controller"]
        
        # 判断是否是JSON格式的任务规范
        if config_content.strip().startswith("{"):
            # Case A: 传入的是 OSWorld 任务规范 (JSON)
            try:
                task_spec = json.loads(config_content)
                setup_steps = task_spec.get("config", [])
                evaluator = task_spec.get("evaluator", {})
                
                # 执行 config 中的每一步 (download, execute 等)
                if setup_steps:
                    from src.utils.desktop_env.controllers.setup import execute_setup_steps
                    execute_setup_steps(controller, setup_steps)
                
                # 将 evaluator 缓存到 GLOBAL_SESSIONS 中供后续 evaluate_task 使用
                GLOBAL_SESSIONS[worker_id]["evaluator"] = evaluator
                
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON in init_script: {e}")
        else:
            # Case B: 传入的是纯 Python 脚本 (如 Math/Web 任务)
            controller.execute_python_command(config_content)
            
        return True
    except Exception as e:
        print(f"VM initialization failed for worker {worker_id}: {e}")
        return False

def _get_controller(worker_id: str) -> PythonController:
    session = GLOBAL_SESSIONS.get(worker_id)
    if not session or not session.get("controller"):
        raise RuntimeError(f"Session not found for worker: {worker_id}. Call 'setup_vm_session' first.")
    return session["controller"]

# --- 生命周期工具 (Group: computer_lifecycle) ---

@ToolRegistry.register_tool("computer_lifecycle")
async def setup_vm_session(config_name: str, task_id: str, worker_id: str, init_script: str = "") -> str:
    """
    初始化 VM 会话：直接申请 VM 资源并初始化控制器。
    
    Args:
        config_name: 配置名称，用于决定申请哪种类型的 VM 资源。
                     - 包含 "computer_13" -> 申请 "vm_computer_13"
                     - 包含 "pyautogui" -> 申请 "vm_pyautogui"
                     - 其他 -> 默认为 "vm_pyautogui"
        task_id: 任务 ID
        worker_id: Worker ID
        init_script: 初始化脚本内容
    """
    
    # 1. [核心修改] 动态资源类型选择
    # 逻辑：根据 config_name 推断 deployment_config.json 中定义的资源 Key
    target_resource_type = "vm_pyautogui"  # 默认值 (因为您的配置中它是 enabled=true)
    
    if config_name:
        cn_lower = config_name.lower()
        if "computer_13" in cn_lower or "computer13" in cn_lower:
            target_resource_type = "vm_computer_13"
        elif "pyautogui" in cn_lower:
            target_resource_type = "vm_pyautogui"
    
    # 2. [核心修改] 设置长超时，允许排队
    # Task 执行和 Reset 较慢，给予 600秒 (10分钟) 的排队等待窗口
    req_timeout = 600.0 

    async with httpx.AsyncClient() as client:
        try:
            # 3. [核心修改] 直接发起申请 (无探活)
            resp = await client.post(
                f"{RESOURCE_API_URL}/allocate",
                json={
                    "worker_id": worker_id, 
                    "type": target_resource_type, # 动态传递资源类型
                    "timeout": req_timeout        # 传递超时参数给服务端
                },
                # HTTP 连接超时需略大于逻辑超时，防止断连
                timeout=req_timeout + 5 
            )
            resp.raise_for_status()
            data = resp.json()
            
        except httpx.TimeoutException:
            # 捕获超时：说明在服务端排队 600s 后仍无资源释放
            return json.dumps({
                "status": "error", 
                "message": f"System busy: Could not acquire '{target_resource_type}' within {req_timeout}s. Resource queue timeout."
            })
        except httpx.HTTPStatusError as e:
            # 捕获 503 等服务端明确返回的错误
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

# =============================================================================
# [核心改造] 动作执行与观测捕获的统一封装
# =============================================================================

async def _execute_and_capture(worker_id: str, action_logic: Callable) -> List[Union[TextContent, ImageContent]]:
    """
    执行动作逻辑，并立即捕获当前屏幕状态和 A11y Tree。
    返回符合 MCP 协议的多模态内容列表。
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
        # 如果动作执行失败，返回错误文本，通常不需要截图（或者也可以截图用于调试）
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
        # 我们将其包装在 XML 标签中，方便 Agent 区分这是 Tree 而不是普通文本
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
# 动作工具 (重构为返回多模态列表)
# =============================================================================

# 1. Computer 13 专属动作
@ToolRegistry.register_tool("desktop_action_computer13")
async def desktop_mouse_move(worker_id: str, x: Optional[int] = None, y: Optional[int] = None) -> list:
    ctrl = _get_controller(worker_id)
    params = {}
    if x is not None and y is not None:
        params = {"x": x, "y": y}
    
    return await _execute_and_capture(worker_id, lambda: 
        ctrl.execute_action({"action_type": "MOVE_TO", "parameters": params})
    )

@ToolRegistry.register_tool("desktop_action_computer13")
async def desktop_mouse_click(worker_id: str, x: Optional[int] = None, y: Optional[int] = None, button: str = "left", num_clicks: int = 1) -> list:
    ctrl = _get_controller(worker_id)
    params = {"button": button, "num_clicks": num_clicks}
    if x is not None and y is not None:
        params.update({"x": x, "y": y})
    
    return await _execute_and_capture(worker_id, lambda: 
        ctrl.execute_action({"action_type": "CLICK", "parameters": params})
    )

@ToolRegistry.register_tool("desktop_action_computer13")
async def desktop_mouse_right_click(worker_id: str, x: Optional[int] = None, y: Optional[int] = None) -> list:
    ctrl = _get_controller(worker_id)
    params = {}
    if x is not None and y is not None:
        params = {"x": x, "y": y}
    
    return await _execute_and_capture(worker_id, lambda: 
        ctrl.execute_action({"action_type": "RIGHT_CLICK", "parameters": params})
    )

@ToolRegistry.register_tool("desktop_action_computer13")
async def desktop_mouse_double_click(worker_id: str, x: Optional[int] = None, y: Optional[int] = None) -> list:
    ctrl = _get_controller(worker_id)
    params = {}
    if x is not None and y is not None:
        params = {"x": x, "y": y}
    
    return await _execute_and_capture(worker_id, lambda: 
        ctrl.execute_action({"action_type": "DOUBLE_CLICK", "parameters": params})
    )

@ToolRegistry.register_tool("desktop_action_computer13")
async def desktop_mouse_drag(worker_id: str, x: int, y: int) -> list:
    ctrl = _get_controller(worker_id)
    return await _execute_and_capture(worker_id, lambda: 
        ctrl.execute_action({"action_type": "DRAG_TO", "parameters": {"x": x, "y": y}})
    )

@ToolRegistry.register_tool("desktop_action_computer13")
async def desktop_scroll(worker_id: str, dx: Optional[int] = None, dy: Optional[int] = None) -> list:
    ctrl = _get_controller(worker_id)
    params = {}
    if dx is not None: params["dx"] = dx
    if dy is not None: params["dy"] = dy
    
    return await _execute_and_capture(worker_id, lambda: 
        ctrl.execute_action({"action_type": "SCROLL", "parameters": params})
    )

@ToolRegistry.register_tool("desktop_action_computer13")
async def desktop_type(worker_id: str, text: str) -> list:
    ctrl = _get_controller(worker_id)
    return await _execute_and_capture(worker_id, lambda: 
        ctrl.execute_action({"action_type": "TYPING", "parameters": {"text": text}})
    )

@ToolRegistry.register_tool("desktop_action_computer13")
async def desktop_key_press(worker_id: str, key: str) -> list:
    ctrl = _get_controller(worker_id)
    return await _execute_and_capture(worker_id, lambda: 
        ctrl.execute_action({"action_type": "PRESS", "parameters": {"key": key}})
    )

@ToolRegistry.register_tool("desktop_action_computer13")
async def desktop_key_hold(worker_id: str, key: str, action: str) -> list:
    ctrl = _get_controller(worker_id)
    act_type = "KEY_DOWN" if action.lower() == "down" else "KEY_UP"
    return await _execute_and_capture(worker_id, lambda: 
        ctrl.execute_action({"action_type": act_type, "parameters": {"key": key}})
    )

@ToolRegistry.register_tool("desktop_action_computer13")
async def desktop_hotkey(worker_id: str, keys: List[str]) -> list:
    ctrl = _get_controller(worker_id)
    return await _execute_and_capture(worker_id, lambda: 
        ctrl.execute_action({"action_type": "HOTKEY", "parameters": {"keys": keys}})
    )

# 2. PyAutoGUI 专属动作
@ToolRegistry.register_tool("desktop_action_pyautogui")
async def desktop_execute_python_script(worker_id: str, script: str) -> list:
    ctrl = _get_controller(worker_id)
    # execute_python_command 可能返回 dict 或 str
    return await _execute_and_capture(worker_id, lambda: 
        ctrl.execute_python_command(script)
    )

# 3. 共享动作 (注册到两个组)
@ToolRegistry.register_tool("desktop_action_computer13")
@ToolRegistry.register_tool("desktop_action_pyautogui")
async def desktop_mouse_button(worker_id: str, action: str, button: str = "left") -> list:
    ctrl = _get_controller(worker_id)
    act_type = "MOUSE_DOWN" if action.lower() == "down" else "MOUSE_UP"
    return await _execute_and_capture(worker_id, lambda: 
        ctrl.execute_action({"action_type": act_type, "parameters": {"button": button}})
    )

@ToolRegistry.register_tool("desktop_action_computer13")
@ToolRegistry.register_tool("desktop_action_pyautogui")
async def desktop_control(worker_id: str, action: str) -> list:
    ctrl = _get_controller(worker_id)
    act_str = action.upper()
    return await _execute_and_capture(worker_id, lambda: 
        ctrl.execute_action(act_str)
    )