#!/usr/bin/env python3
"""
OSWorld 任务执行演示 (基于 demo_single_resource_flow.py)

功能：
1. 分配 vm_pyautogui 资源
2. 解析 OSWorld 任务格式 (config) 并自动执行环境配置 (Setup)
3. 执行任务目标 (统计 PHP 文件行数)
4. 释放资源
"""

import os
import sys
import json
import logging
import asyncio

# 假设脚本在项目根目录或合适的位置，确保可以导入 utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 尝试导入 MCPSSEClient，如果环境未设置，需确保路径正确
try:
    from utils.mcp_sse_client import MCPSSEClient
except ImportError:
    # 简单的 mock 或提示，实际运行时需确保环境正确
    logging.warning("Could not import MCPSSEClient. Please run this script in the correct environment.")
    class MCPSSEClient:
        def __init__(self, url): pass
        async def connect(self): pass
        async def list_tools(self): return []
        async def call_tool(self, name, args): return type('obj', (object,), {'content': []})

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==========================================
# 输入数据 (Task Data)
# ==========================================
TASK_DATA = {
  "id": "4127319a-8b79-4410-b58a-7a151e15f3d7",
  "question": "Use terminal command to count all the lines of all php files in current directory recursively, show the result on the terminal",
  "answer": "",
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
        "command": "chmod +x setup.sh",
        "shell": True
      }
    },
    {
      "type": "execute",
      "parameters": {
        "command": "bash ./setup.sh",
        "shell": True
      }
    },
    {
      "type": "execute",
      "parameters": {
        "command": "export DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus'\nxdg-open /home/user/project",
        "shell": True
      }
    }
  ],
  "evaluator": {
    "func": "check_include_exclude",
    "result": {
      "type": "vm_terminal_output"
    },
    "expected": {
      "type": "rule",
      "rules": {
        "include": [
          "54"
        ],
        "exclude": []
      }
    }
  }
}

class OSWorldTaskRunner:
    """
    OSWorld 任务执行器
    """

    def __init__(self, server_url: str = "http://localhost:8080"):
        self.server_url = server_url
        self.worker_id = "task_runner_001"
        self.mcp_client = MCPSSEClient(f"{server_url}/sse")
        self.allocated_resource_id = None
        self.resource_type = "vm_pyautogui"

        logger.info(f"[{self.worker_id}] Initialized with server: {server_url}")

    async def connect(self):
        logger.info(f"[{self.worker_id}] Connecting to MCP Server...")
        await self.mcp_client.connect()
        logger.info(f"[{self.worker_id}] ✅ Connected")

    async def allocate_resource(self):
        """分配资源"""
        logger.info(f"[{self.worker_id}] Allocating {self.resource_type}...")
        try:
            result = await self.mcp_client.call_tool(
                "allocate_single_resource",
                {
                    "worker_id": self.worker_id,
                    "resource_type": self.resource_type,
                    "timeout": 600
                }
            )
            response = self._parse_mcp_response(result)
            
            # 检查错误
            if response.get("status") == "error":
                logger.error(f"Allocation failed: {response}")
                return False
                
            # 获取资源ID
            resource_info = response.get(self.resource_type)
            if resource_info:
                self.allocated_resource_id = resource_info.get("id")
                logger.info(f"[{self.worker_id}] ✅ Allocated Resource ID: {self.allocated_resource_id}")
                
                # 初始化资源 (Setup Batch)
                await self._initial_setup(response)
                return True
            return False
        except Exception as e:
            logger.error(f"Allocation error: {e}")
            return False

    async def _initial_setup(self, allocation_data):
        """调用 setup_batch_resources 进行基础初始化"""
        logger.info(f"[{self.worker_id}] Performing base resource setup...")
        await self.mcp_client.call_tool(
            "setup_batch_resources",
            {
                "worker_id": self.worker_id,
                "resource_init_configs": {}, # 这里可以传递基础快照配置
                "allocated_resources": allocation_data
            }
        )

    async def run_task_setup(self, config_steps):
        """
        执行 Task 中的 config 步骤 (下载文件, 运行命令)
        """
        logger.info(f"[{self.worker_id}] 🚀 Starting Task Environment Setup...")
        
        for i, step in enumerate(config_steps):
            step_type = step.get("type")
            params = step.get("parameters", {})
            logger.info(f"[{self.worker_id}] Executing Step {i+1}: {step_type}")

            if step_type == "download":
                files = params.get("files", [])
                for f in files:
                    url = f.get("url")
                    path = f.get("path")
                    # 构造 wget 命令下载文件
                    cmd = f"wget -O {path} {url}"
                    logger.info(f"  Downloading: {url} -> {path}")
                    await self._execute_shell_command(cmd)

            elif step_type == "execute":
                cmd = params.get("command")
                logger.info(f"  Executing Command: {cmd}")
                await self._execute_shell_command(cmd)
            
            else:
                logger.warning(f"  Unknown step type: {step_type}")

        logger.info(f"[{self.worker_id}] ✅ Task Environment Setup Completed.")

    async def run_agent_task(self, question):
        """
        模拟 Agent 执行任务
        """
        logger.info(f"[{self.worker_id}] 🤖 Simulating Agent Action for question: {question}")
        
        # 针对题目 "count all the lines of all php files in current directory recursively"
        # 构造解决方案命令
        # 注意：这里模拟 Agent 思考后生成的命令
        solution_command = "find . -name '*.php' -type f -print0 | xargs -0 wc -l"
        
        logger.info(f"[{self.worker_id}] Agent decided to run: {solution_command}")
        
        output = await self._execute_shell_command(solution_command)
        
        logger.info(f"[{self.worker_id}] 📄 Command Output:\n{output}")
        
        # 简单的验证（根据 evaluator.expected.rules.include）
        expected_output = "54"
        if expected_output in output:
            logger.info(f"[{self.worker_id}] ✅ Verification Passed: Output contains '{expected_output}'")
        else:
            logger.warning(f"[{self.worker_id}] ⚠️ Verification Warning: Expected '{expected_output}' not found explicitly.")

    async def _execute_shell_command(self, command):
        """
        调用工具执行 Shell 命令
        注意：这里假设 Server 端有一个 'execute_command' 或 'run_terminal_cmd' 工具
        """
        try:
            # 尝试调用 execute_command (通用名称)
            # 如果您的环境中工具名不同（如 'computer' 工具的 'terminal' 动作），请修改此处
            result = await self.mcp_client.call_tool(
                "execute_command", 
                {
                    "worker_id": self.worker_id,
                    "command": command
                }
            )
            
            # 解析文本结果
            output = ""
            if hasattr(result, 'content'):
                for item in result.content:
                    if item.type == 'text':
                        output += item.text
            
            return output
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return f"Error: {e}"

    async def release(self):
        """释放资源"""
        if self.allocated_resource_id:
            logger.info(f"[{self.worker_id}] Releasing resource {self.allocated_resource_id}...")
            await self.mcp_client.call_tool(
                "release_batch_resources",
                {
                    "worker_id": self.worker_id,
                    "resource_ids": [self.allocated_resource_id]
                }
            )
            logger.info(f"[{self.worker_id}] ✅ Resource released")
        await self.mcp_client.disconnect()

    def _parse_mcp_response(self, response):
        """简单的 JSON 解析帮助函数"""
        try:
            if hasattr(response, 'content') and response.content:
                text = response.content[0].text
                return json.loads(text)
        except:
            pass
        return {}

async def main():
    # 从环境变量或默认值获取 Server URL
    server_url = os.environ.get("MCP_SERVER_URL", "http://localhost:8080")
    
    runner = OSWorldTaskRunner(server_url)
    
    try:
        # 1. 连接
        await runner.connect()
        
        # 2. 分配资源
        if not await runner.allocate_resource():
            logger.error("Failed to allocate resource. Exiting.")
            return

        # 3. 配置环境 (基于 JSON 中的 config)
        await runner.run_task_setup(TASK_DATA["config"])
        
        # 4. 执行任务 (基于 JSON 中的 question)
        await runner.run_agent_task(TASK_DATA["question"])

    except Exception as e:
        logger.error(f"Runtime error: {e}", exc_info=True)
    finally:
        # 5. 释放资源
        await runner.release()

if __name__ == "__main__":
    asyncio.run(main())