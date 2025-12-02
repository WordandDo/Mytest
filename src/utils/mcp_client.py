# src/utils/mcp_client.py
import os
import json
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

    async def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        """
        调用工具并返回解析后的文本结果。
        
        :param name: 工具名称
        :param arguments: 工具参数字典
        :return: 工具执行结果 (字符串)
        """
        if not self.session:
            raise RuntimeError("MCP Client not connected!")
        
        if arguments is None:
            arguments = {}

        # === [新增日志 START] ===
        # 打印请求详情 (截断过长的参数，如 init_script)
        debug_args = arguments.copy()
        for k, v in debug_args.items():
            if isinstance(v, str) and len(v) > 200:
                debug_args[k] = v[:200] + "...(truncated)"
        print(f"\n[MCP-CLI] ➡️ REQ Tool: {name}")
        # ensure_ascii=False 允许打印中文
        print(f"[MCP-CLI]    Args: {json.dumps(debug_args, ensure_ascii=False)}")
        # === [新增日志 END] ===

        # 发送请求
        # 注意：这里移除了 explicit type hint 'CallToolResult' 以避免额外的 import 错误，
        # 如果你有导入它，可以加回去: result: CallToolResult = ...
        result = await self.session.call_tool(name, arguments)
        
        # === [新增日志 START] ===
        # 打印响应摘要
        content_summary = "Empty"
        if result.content:
            # 简单转字符串预览，防止大量 Base64 图片刷屏
            content_summary = str(result.content)[:500] 
        print(f"[MCP-CLI] ⬅️ RES Tool: {name}")
        print(f"[MCP-CLI]    Data: {content_summary}\n")
        # === [新增日志 END] ===

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