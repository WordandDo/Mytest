# src/utils/mcp_client.py
import os
import asyncio
from contextlib import AsyncExitStack
from typing import Optional, Dict, Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MCPClientWrapper:
    """
    通用 MCP 客户端封装器。
    负责启动 Server 子进程并维护 Session。
    """
    def __init__(self, server_script_path: str):
        self.server_script_path = server_script_path
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    async def connect(self):
        """启动子进程并建立连接"""
        # 使用当前环境的 Python 解释器启动 Server
        server_params = StdioServerParameters(
            command="python",
            args=[self.server_script_path],
            env=os.environ.copy()  # 传递环境变量(如 OPENAI_API_KEY)
        )

        # 1. 建立传输层 (Transport)
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.read, self.write = stdio_transport

        # 2. 建立会话层 (Session)
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.read, self.write)
        )
        
        # 3. 初始化协议
        await self.session.initialize()

    async def call_tool(self, name: str, arguments: Dict[str, Any] = None) -> Any:
        """调用工具并提取文本结果"""
        if not self.session:
            raise RuntimeError("MCP Client not connected!")
        
        if arguments is None:
            arguments = {}

        # 发送请求
        result = await self.session.call_tool(name, arguments)
        
        # 解析标准 MCP 响应 (Content List)
        output_text = []
        if result.content:
            for item in result.content:
                if item.type == 'text':
                    output_text.append(item.text)
                elif item.type == 'image':
                    # 如果有图片，可能需要特殊处理，这里暂转为描述
                    output_text.append("[Image Data]")
        
        return "\n".join(output_text) if output_text else "Success"

    async def close(self):
        """关闭连接和子进程"""
        await self.exit_stack.aclose()