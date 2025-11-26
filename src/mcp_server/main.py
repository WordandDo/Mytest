# src/mcp_server/main.py
import asyncio
import json
import httpx
import sys
import os
from typing import Any
import base64
# 确保能导入项目根目录的模块
sys.path.append(os.getcwd())

from mcp.server.fastmcp import FastMCP
from src.utils.desktop_env.controllers.python import PythonController

# 初始化 MCP Server
mcp = FastMCP("OSWorld Gateway")

# 配置：资源服务地址 (Layer 3)
# 注意：确保这个地址与您启动 resource_api.py 时的地址一致
RESOURCE_API_URL = "http://localhost:8000"

# 全局会话状态 (由于 Stdio 模式下每个 Client 对应一个 Server 进程，这里可以用全局变量)
current_session = {
    "controller": None,  # PythonController 实例
    "env_id": None,      # 资源 ID (如 vm-1)
    "task_id": None      # 当前任务 ID
}

def _get_controller() -> PythonController:
    """获取当前激活的控制器，如果没有则抛错"""
    ctrl = current_session.get("controller")
    if not ctrl:
        raise RuntimeError("Environment not initialized. Call 'setup_environment' first.")
    return ctrl

# ----------------------------------------------------------------
# 1. 生命周期管理工具
# ----------------------------------------------------------------

@mcp.tool()
async def setup_environment(config_name: str, task_id: str) -> str:
    """
    初始化环境：申请 VM 资源并连接。
    返回：初始状态的 JSON 字符串。
    """
    # 1. 向 Resource API 申请资源
    async with httpx.AsyncClient() as client:
        try:
            # worker_id 用 task_id 标识
            resp = await client.post(f"{RESOURCE_API_URL}/allocate", json={"worker_id": task_id})
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return json.dumps({"status": "error", "message": f"Resource allocation failed: {str(e)}"})

    # 解析资源信息
    # 假设 Resource API 返回格式: {"id": "vm_1", "ip": "192.168.x.x", "port": 5000, ...}
    # 注意：您的 SimpleManager.allocate 返回的是 VMPoolImpl 的 allocate 结果
    # 通常包含 'id', 'ip', 'port' 等
    env_id = data.get("id")
    ip = data.get("ip")
    port = data.get("port", 5000)

    if not ip:
        return json.dumps({"status": "error", "message": "Allocated resource has no IP"})

    # 2. 初始化本地控制器 (连接到远程 VM)
    try:
        # 确保 PythonController 支持 vm_ip 参数
        controller = PythonController(vm_ip=ip, server_port=port)
        
        # 3. 更新会话状态
        current_session["controller"] = controller
        current_session["env_id"] = env_id
        current_session["task_id"] = task_id
        
        # 4. 获取初始截图作为验证
        screenshot = controller.get_screenshot()
        return json.dumps({
            "status": "success", 
            "env_id": env_id, 
            "ip": ip,
            "info": "Environment Ready"
        })
    except Exception as e:
        return json.dumps({"status": "error", "message": f"Controller init failed: {str(e)}"})

@mcp.tool()
async def teardown_environment() -> str:
    """释放资源并清理环境"""
    env_id = current_session.get("env_id")
    task_id = current_session.get("task_id")
    
    if not env_id:
        return "No active environment to teardown."

    # 向 Resource API 发送释放请求
    async with httpx.AsyncClient() as client:
        try:
            await client.post(
                f"{RESOURCE_API_URL}/release", 
                json={"resource_id": env_id, "worker_id": task_id or "unknown"}
            )
        except Exception as e:
            return f"Error releasing resource: {str(e)}"
    
    # 清理本地状态
    current_session["controller"] = None
    current_session["env_id"] = None
    return "Environment released successfully."

# ----------------------------------------------------------------
# 2. 桌面操作工具 (Action Tools)
# ----------------------------------------------------------------

@mcp.tool()
async def click(x: int, y: int, button: str = "left") -> str:
    """在指定坐标 (x, y) 点击鼠标。button 可选 'left', 'right', 'middle'。"""
    ctrl = _get_controller()
    action = {"action": "click", "coordinate": [x, y], "button": button}
    ctrl.execute_action(action)
    return json.dumps({"status": "success", "action": "click", "args": [x, y, button]})

@mcp.tool()
async def type_text(text: str) -> str:
    """在当前焦点处输入文本。"""
    ctrl = _get_controller()
    action = {"action": "type", "text": text}
    ctrl.execute_action(action)
    return json.dumps({"status": "success", "action": "type", "text": text})

@mcp.tool()
async def scroll(delta_x: int, delta_y: int) -> str:
    """滚动鼠标滚轮。"""
    ctrl = _get_controller()
    action = {"action": "scroll", "offset": [delta_x, delta_y]}
    ctrl.execute_action(action)
    return json.dumps({"status": "success", "action": "scroll"})

@mcp.tool()
async def key_press(key: str) -> str:
    """按下一个按键 (如 'enter', 'space', 'backspace', 'ctrl+c')。"""
    ctrl = _get_controller()
    action = {"action": "key", "text": key}
    ctrl.execute_action(action)
    return json.dumps({"status": "success", "action": "key_press", "key": key})

# ----------------------------------------------------------------
# 3. 感知工具 (Observation Tools)
# ----------------------------------------------------------------


@mcp.tool()
async def get_observation() -> str:
    """获取当前屏幕状态（截图和 Accessibility Tree）。"""
    ctrl = _get_controller()
    
    # 获取原始数据
    screenshot_bytes = ctrl.get_screenshot()  # 这是 bytes
    a11y_tree = ctrl.get_accessibility_tree() # 这是 str
    
    # [核心修复] 将 bytes 转为 Base64 字符串
    if screenshot_bytes:
        screenshot_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')
    else:
        screenshot_base64 = ""
    
    return json.dumps({
        "screenshot": screenshot_base64, # 修复后这里是 str，可以被序列化
        "accessibility_tree": a11y_tree
    })

@mcp.tool()
async def evaluate_task() -> str:
    """
    [可选] 计算当前任务的得分。
    需要在 Server 端集成 evaluators 逻辑。
    """
    # 示例逻辑：总是返回 0.0，您需要根据实际 metrics 实现
    return "0.0"

if __name__ == "__main__":
    # 启动 Server
    mcp.run()