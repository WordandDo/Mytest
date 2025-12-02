# src/utils/mcp_sse_client.py
import asyncio
from contextlib import AsyncExitStack
import json
from typing import Optional, Dict, Any, List

# 引入 MCP SDK 的 SSE 客户端
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.types import CallToolResult, Tool

class MCPSSEClient:
    """
    兼容 SSE (Server-Sent Events) 协议的 MCP 客户端。
    用于连接通过 HTTP 暴露的 MCP 网关 (如 start_gateway.sh 启动的服务)。
    """
    def __init__(self, server_url: str = "http://localhost:8080/sse"):
        """
        :param server_url: SSE 端点地址 (例如 http://localhost:8080/sse)
        """
        self.server_url = server_url
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    async def connect(self):
        """建立 SSE 连接并初始化会话"""
        print(f"📡 Connecting to SSE Endpoint: {self.server_url}...")
        
        try:
            # 1. 建立传输层 (Transport)
            # sse_client context manager 返回 (read_stream, write_stream)
            streams = await self.exit_stack.enter_async_context(sse_client(self.server_url))
            self.read, self.write = streams

            # 2. 建立会话层 (Session)
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(self.read, self.write)
            )
            
            # 3. 初始化协议 (发送 InitializeRequest)
            await self.session.initialize()
            print("✅ MCP Session Initialized")
            
        except Exception as e:
            print(f"❌ Connection Failed: {e}")
            raise e

    async def list_tools(self) -> List[Tool]:
        """获取服务器暴露的所有工具列表"""
        if not self.session:
            raise RuntimeError("Client not connected. Call connect() first.")
        
        result = await self.session.list_tools()
        return result.tools

    async def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        """
        调用工具并返回解析后的文本结果。
        
        :param name: 工具名称
        :param arguments: 工具参数字典
        :return: 工具执行结果 (字符串)
        """
        if not self.session:
            raise RuntimeError("Client not connected. Call connect() first.")
        
        if arguments is None:
            arguments = {}
            
        # === [新增日志 START] ===
        # 打印请求详情 (截断过长的参数，如 init_script)
        debug_args = arguments.copy()
        for k, v in debug_args.items():
            if isinstance(v, str) and len(v) > 200:
                debug_args[k] = v[:200] + "...(truncated)"
        print(f"\n[MCP-CLI] ➡️ REQ Tool: {name}")
        print(f"[MCP-CLI]    Args: {json.dumps(debug_args, ensure_ascii=False)}")
        # === [新增日志 END] ===

        result = await self.session.call_tool(name, arguments)
        
        # === [新增日志 START] ===
        # 打印响应摘要
        content_summary = "Empty"
        if result.content:
            # 限制日志长度
            content_summary = str(result.content)[:500]
        print(f"[MCP-CLI] ⬅️ RES Tool: {name}")
        print(f"[MCP-CLI]    Data: {content_summary}\n")
        # === [新增日志 END] ===
        
        # 解析结果 (MCP 可以返回 Text 或 Image)
        output_parts = []
        if result.content:
            for item in result.content:
                if item.type == 'text':
                    output_parts.append(item.text)
                elif item.type == 'image':
                    output_parts.append(f"[Image: {item.mimeType}]")
                elif item.type == 'resource':
                     # 修复：通过.resource属性访问uri
                     output_parts.append(f"[Resource: {item.resource.uri}]")

        # 如果没有内容，可能是执行成功但无返回
        return "\n".join(output_parts) if output_parts else "Success (No output)"

    async def close(self):
        """优雅关闭连接"""
        await self.exit_stack.aclose()
        print("🔌 Client Disconnected")

# --- 简单的自测代码 ---
async def main():
    # 假设网关已在 8080 启动
    client = MCPSSEClient("http://localhost:8080/sse")
    try:
        await client.connect()
        
        # 1. 列出工具
        tools = await client.list_tools()
        print(f"\n🔍 Found {len(tools)} tools:")
        for t in tools:
            print(f"   - {t.name}: {t.description[:50] if t.description else ''}...")
            
        # 2. (可选) 尝试调用一个简单工具，例如 evaluate_task
        # print("\n▶️  Calling evaluate_task...")
        # res = await client.call_tool("evaluate_task", {"worker_id": "test-sse"})
        # print(f"   Result: {res}")

    except Exception as e:
        print(f"Test Error: {e}")
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())