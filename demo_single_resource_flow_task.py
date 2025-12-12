#!/usr/bin/env python3
"""
OSWorld 任务执行演示 (VM PyAutoGUI 生产环境完整版)

此脚本演示了如何使用 MCP (Model Context Protocol) Client 与 Server 进行交互，
完成从资源申请、环境初始化、任务执行到资源释放的全流程。

前置条件：
1. 确保 MCP Server (Gateway) 已在 http://localhost:8080 启动。
2. 确保项目根目录下包含 src/utils 模块。
"""

#!/usr/bin/env python3
"""
OSWorld 任务执行演示 (VM PyAutoGUI 生产环境完整版 - 智能日志优化)

此脚本演示了如何使用 MCP (Model Context Protocol) Client 与 Server 进行交互。
包含自动资源管理、安全白名单及智能日志清洗功能。

前置条件：
1. 确保 MCP Server (Gateway) 已在 http://localhost:8080 启动。
2. 确保项目根目录下包含 src/utils 模块。
"""

import os
import sys
import json
import logging
import asyncio

# ==========================================
# 1. 环境路径配置
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将 'src' 目录添加到 Python 搜索路径，以便导入 utils
sys.path.insert(0, os.path.join(current_dir, "src"))

# ==========================================
# 2. 核心依赖导入
# ==========================================
try:
    from utils.mcp_sse_client import MCPSSEClient
    from mcp.types import Tool
except ImportError as e:
    logging.critical(f"❌ 依赖导入失败: {e}")
    logging.critical("请确保在正确的项目根目录下运行，且 'src/utils' 存在。")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TaskRunner")

# ==========================================
# 3. 静态配置
# ==========================================

# [配置] 工具白名单
ALLOWED_TOOL_GROUPS = {
    "pyautogui_lifecycle", 
    "pyautogui_observation", 
    "desktop_action_pyautogui"
}

# [数据] 任务定义
TASK_DATA = {
  "id": "4127319a-8b79-4410-b58a-7a151e15f3d7",
  "question": "Use terminal command to count all the lines of all php files in current directory recursively, show the result on the terminal",
  "config": [
    {
      "type": "download",
      "parameters": {
        "files": [
          {
            "url": "https://hf-mirror.com/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/os/4127319a-8b79-4410-b58a-7a151e15f3d7/setup.sh",
            "path": "setup.sh"
          }
        ]
      }
    },
    {
      "type": "execute",
      "parameters": {
        "command": "mkdir -p /home/user/Desktop && echo 'INIT LOG START' > /home/user/Desktop/init_steps.log",
        "shell": True
      }
    },
    {
      "type": "execute",
      "parameters": {"command": "chmod +x setup.sh", "shell": True}
    },
    {
      "type": "execute",
      "parameters": {
        "command": "if [ -f setup.sh ]; then echo 'STEP 1: setup.sh exists after download' >> /home/user/Desktop/init_steps.log; else echo 'STEP 1: setup.sh missing' >> /home/user/Desktop/init_steps.log; fi",
        "shell": True
      }
    },
    {
      "type": "execute",
      "parameters": {
        "command": "if [ -x setup.sh ]; then echo 'STEP 2: setup.sh is executable' >> /home/user/Desktop/init_steps.log; else echo 'STEP 2: setup.sh not executable' >> /home/user/Desktop/init_steps.log; fi",
        "shell": True
      }
    },
    {
      "type": "execute",
      "parameters": {
        "command": "if bash ./setup.sh; then echo 'STEP 3: setup.sh executed successfully' >> /home/user/Desktop/init_steps.log; else echo 'STEP 3: setup.sh execution failed' >> /home/user/Desktop/init_steps.log; exit 1; fi",
        "shell": True
      }
    },
    {
      "type": "execute",
      "parameters": {
        "command": "mkdir -p /home/user/Desktop && cp setup.sh /home/user/Desktop/setup.sh && echo 'STEP 4: artifacts copied to desktop' >> /home/user/Desktop/init_steps.log && echo '初始化完成，请在桌面确认 setup.sh 文件和 init_ready.txt' > /home/user/Desktop/init_ready.txt && ls -l setup.sh | tee -a /home/user/Desktop/init_steps.log",
        "shell": True
      }
    },
    {
      "type": "execute",
      "parameters": {
        "command": "export DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus'\nxdg-open /home/user/project\n echo 'STEP 5: desktop open command issued' >> /home/user/Desktop/init_steps.log",
        "shell": True
      }
    }
  ]
}

# ==========================================
# 4. 任务执行器类
# ==========================================

class OSWorldPyAutoGUIRunner:
    def __init__(self, server_url: str = "http://localhost:8080", worker_id: str = "task_runner_prod_001"):
        self.server_url = server_url
        self.worker_id = worker_id
        self.mcp_client = MCPSSEClient(f"{server_url}/sse")
        self.initialized = False
        self.agent_tools = [] 

        logger.info(f"[{self.worker_id}] 初始化完成，目标服务器: {server_url}")

    async def connect(self):
        logger.info(f"[{self.worker_id}] 正在连接 MCP Server...")
        await self.mcp_client.connect()
        logger.info(f"[{self.worker_id}] ✅ 连接成功")

    async def fetch_and_filter_tools(self):
        logger.info(f"[{self.worker_id}] 获取并过滤工具列表...")
        try:
            all_tools = await self.mcp_client.list_tools()
            self.agent_tools = []
            
            for tool in all_tools:
                name = tool.name
                group = getattr(tool, "group", None) 
                
                if not group and hasattr(tool, "metadata") and tool.metadata:
                    group = tool.metadata.get("group")

                if group and group in ALLOWED_TOOL_GROUPS:
                    self.agent_tools.append(tool)
                elif ("pyautogui" in name or "desktop_" in name):
                    self.agent_tools.append(tool)
            
            logger.info(f"[{self.worker_id}] 🛡️ 白名单应用完成。可用工具数: {len(self.agent_tools)}")
            return True
        except Exception as e:
            logger.error(f"获取工具失败: {e}", exc_info=True)
            raise e

    async def setup_session(self):
        logger.info(f"[{self.worker_id}] 开始建立会话 (资源分配 + 自动化配置)...")
        
        init_script_json = json.dumps(TASK_DATA)
        
        try:
            result = await self.mcp_client.call_tool(
                "setup_pyautogui_session",
                {
                    "config_name": "auto_task",
                    "task_id": TASK_DATA["id"],
                    "worker_id": self.worker_id,
                    "init_script": init_script_json
                }
            )
            
            response = self._parse_mcp_response(result)
            
            # [日志优化] 打印清洗后的响应结构，隐藏超长内容
            sanitized_resp = self._sanitize_log_data(response)
            logger.info(f"[{self.worker_id}] Setup 响应详情:\n{json.dumps(sanitized_resp, indent=2, ensure_ascii=False)}")

            if response.get("status") == "error":
                error_msg = response.get('message', 'Unknown error')
                logger.error(f"[{self.worker_id}] 初始化失败: {error_msg}")
                raise RuntimeError(f"Session setup failed: {error_msg}")
            
            self.initialized = True
            logger.info(f"[{self.worker_id}] ✅ 会话建立成功")
            return True
            
        except Exception as e:
            logger.error(f"[{self.worker_id}] 初始化异常: {e}")
            raise e

    async def run_agent_task(self):
        question = TASK_DATA["question"]
        logger.info(f"[{self.worker_id}] 🤖 Agent 收到问题: {question}")
        
        tool_to_use = "desktop_execute_python_script"
        allowed_names = [t.name for t in self.agent_tools]
        if tool_to_use not in allowed_names:
            logger.error(f"🚨 安全警报: 试图调用未授权工具 '{tool_to_use}'")
            return

        solution_shell_cmd = "find . -name '*.php' -type f -print0 | xargs -0 wc -l"
        logger.info(f"[{self.worker_id}] Agent 决定执行命令: {solution_shell_cmd}")
        
        output = await self._execute_shell_command(solution_shell_cmd)
        
        # [日志优化] 如果输出过长，进行截断显示
        log_output = self._sanitize_log_data(output)
        logger.info(f"[{self.worker_id}] 📄 命令输出:\n{log_output}")
        
        expected_output = "54"
        if expected_output in output:
            logger.info(f"[{self.worker_id}] ✅ 结果验证通过")
        else:
            logger.warning(f"[{self.worker_id}] ⚠️ 结果验证未通过")

    async def hold_desktop_for_observation(self, duration_seconds: int = 300):
        """
        Keep the desktop open to let operators verify the initialization state.
        """
        logger.info(f"[{self.worker_id}] ⏳ 保持桌面约 {duration_seconds}s 以观察初始化状态...")
        hold_script = f"""import time
print('Observation window started for {duration_seconds} seconds...')
time.sleep({duration_seconds})
print('Observation window ended.')
"""
        await self._execute_desktop_python_script(hold_script, description="initialization observation")

    async def _execute_shell_command(self, command):
        safe_command = command.replace("'", "\\'")
        python_wrapper = f"""
import subprocess
try:
    cmd = '{safe_command}'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
except Exception as e:
    print(f"Execution Error: {{e}}")
"""
        try:
            result = await self.mcp_client.call_tool(
                "desktop_execute_python_script", 
                {
                    "worker_id": self.worker_id,
                    "script": python_wrapper
                }
            )
            
            output = ""
            if hasattr(result, 'content'):
                for item in result.content:
                    if item.type == 'text':
                        output += item.text
            return output
        except Exception as e:
            logger.error(f"命令执行失败: {e}")
            return f"Error: {e}"

    async def _execute_desktop_python_script(self, script: str, description: str = "desktop python script"):
        try:
            result = await self.mcp_client.call_tool(
                "desktop_execute_python_script",
                {
                    "worker_id": self.worker_id,
                    "script": script
                }
            )
            output = ""
            if hasattr(result, 'content'):
                for item in result.content:
                    if item.type == 'text':
                        output += item.text
            log_output = self._sanitize_log_data(output)
            logger.info(f"[{self.worker_id}] {description} 输出:\n{log_output}")
            return output
        except Exception as e:
            logger.error(f"[{self.worker_id}] 执行 {description} 失败: {e}", exc_info=True)
            return f"Error: {e}"

    async def perform_visible_python_action(self):
        logger.info(f"[{self.worker_id}] 🎬 执行第二次初始化的 Python 可视化动作...")
        action_script = """from pathlib import Path
from datetime import datetime

desktop = Path.home() / "Desktop"
desktop.mkdir(parents=True, exist_ok=True)
marker = desktop / "second_init_python_action.txt"
timestamp = datetime.now().isoformat()
marker.write_text(f"✅ Second initialization Python action logged at {timestamp}\\n")
print(f"Visible action file created at: {marker}")
print(f"File contents: {marker.read_text().strip()}")
"""
        await self._execute_desktop_python_script(action_script, description="visible python action")

    async def release(self):
        if self.initialized:
            logger.info(f"[{self.worker_id}] 正在清理会话资源...")
            try:
                if self.mcp_client.session:
                    await self.mcp_client.call_tool(
                        "teardown_pyautogui_environment",
                        {"worker_id": self.worker_id}
                    )
                    logger.info(f"[{self.worker_id}] ✅ 资源已释放")
            except Exception as e:
                logger.warning(f"资源释放请求失败 (忽略): {e}")
            finally:
                self.initialized = False
                
        try:
            await self.mcp_client.close()
            logger.info(f"[{self.worker_id}] 🔌 客户端已断开")
        except Exception:
            pass

    def _parse_mcp_response(self, result):
        try:
            if hasattr(result, 'content') and result.content:
                text = result.content[0].text
                return json.loads(text)
        except json.JSONDecodeError:
            return {"status": "unknown", "text": text}
        except Exception:
            pass
        return {}

    def _sanitize_log_data(self, data):
        """
        [Helper] 智能清洗数据，将过长的 Base64 图片或 XML 树替换为占位符。
        """
        # 1. 字典递归处理
        if isinstance(data, dict):
            new_dict = {}
            for k, v in data.items():
                # 针对特定字段名直接截断
                if k in ["screenshot", "accessibility_tree", "html", "source"]:
                    new_dict[k] = self._truncate_string(v, max_len=100)
                else:
                    new_dict[k] = self._sanitize_log_data(v)
            return new_dict
        
        # 2. 列表递归处理
        elif isinstance(data, list):
            return [self._sanitize_log_data(i) for i in data]
        
        # 3. 字符串智能检测
        elif isinstance(data, str):
            # 检测 XML 结束标签
            if "</accessibility_tree>" in data:
                return self._truncate_string(data, max_len=200, label="[XML Tree]")
            # 检测 Base64 图片头 (简单判断)
            if data.startswith("iVBORw0KGgo") and len(data) > 500:
                return self._truncate_string(data, max_len=50, label="[Base64 Image]")
            # 普通长文本截断
            if len(data) > 2000:
                return self._truncate_string(data, max_len=500, label="[Long Text]")
            return data
            
        else:
            return data

    def _truncate_string(self, text, max_len=100, label=""):
        """字符串截断辅助函数"""
        if not isinstance(text, str):
            return str(text) # 非字符串直接转存
        if len(text) <= max_len:
            return text
        prefix = f"{label} " if label else ""
        return f"{prefix}{text[:max_len]}... <total {len(text)} chars> ...{text[-20:]}"

# ==========================================
# 5. 初始化阶段执行器
# ==========================================

async def _run_initialization_phase(
    server_url: str,
    worker_id: str,
    *,
    hold_seconds: int = 0,
    run_agent_task: bool = False,
    action_cb=None
):
    runner = OSWorldPyAutoGUIRunner(server_url, worker_id=worker_id)
    try:
        logger.info(
            f"[{worker_id}] 阶段启动 (hold_seconds={hold_seconds}, run_agent_task={run_agent_task})"
        )
        await runner.connect()
        await runner.fetch_and_filter_tools()
        await runner.setup_session()
        if run_agent_task:
            await runner.run_agent_task()
        if hold_seconds:
            await runner.hold_desktop_for_observation(duration_seconds=hold_seconds)
        if action_cb:
            await action_cb(runner)
        logger.info(f"[{worker_id}] 阶段流程完成，准备释放资源以触发快速重置。")
    finally:
        await runner.release()


# ==========================================
# 6. 主程序入口
# ==========================================

async def main():
    server_url = os.environ.get("MCP_SERVER_URL", "http://localhost:8080")
    try:
        await _run_initialization_phase(
            server_url,
            worker_id="task_runner_init_observe",
            hold_seconds=300
        )

        await _run_initialization_phase(
            server_url,
            worker_id="task_runner_action_demo",
            run_agent_task=True,
            action_cb=lambda runner: runner.perform_visible_python_action()
        )
        logger.info("两轮初始化与行为验证完成。")

    except BaseException as e:
        logger.error(f"运行时错误或用户中断: {repr(e)}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
