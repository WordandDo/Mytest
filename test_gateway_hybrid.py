import asyncio
import sys
import os

# 确保能找到模块
sys.path.append(os.getcwd())

# 导入新写好的 SSE Client
from src.utils.mcp_sse_client import MCPSSEClient

async def run_test():
    # 指向 start_gateway.sh 监听的地址
    client = MCPSSEClient("http://localhost:8080/sse")
    
    await client.connect()
    
    # 直接调用工具
    result = await client.call_tool("setup_vm_session", {
        "worker_id": "test-user-1",
        "config_name": "default",
        "task_id": "101"
    })
    print(result)
    
    await client.close()

if __name__ == "__main__":
    asyncio.run(run_test())