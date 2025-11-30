import asyncio
import sys
import os
import uuid

# 添加 src 到路径
sys.path.append(os.path.join(os.getcwd(), "src"))

from utils.mcp_client import MCPClientWrapper

async def run_test():
    # 1. 定义 RAG Server 的路径
    server_script = "src/mcp_server/rag_server.py"
    
    print(f"📡 Connecting to MCP Server: {server_script}...")
    
    # 2. 启动 MCP Client (它会自动启动 rag_server.py 子进程)
    # 注意：确保此时 start_backend.sh 已经在另一个终端运行
    client = MCPClientWrapper(server_script)
    
    try:
        await client.connect()
        print("✅ MCP Connection Established")
        
        # 生成一个测试用的 worker_id
        worker_id = f"test-worker-{uuid.uuid4().hex[:4]}"
        
        # 3. 调用工具: 初始化 Session (rag_lifecycle)
        print(f"\n[1] Allocating RAG Session for {worker_id}...")
        init_result = await client.call_tool("setup_rag_session", {"worker_id": worker_id})
        print(f"Result: {init_result}")
        
        if "error" in init_result:
            print("❌ Setup failed. Is the Resource API running?")
            return

        # 4. 调用工具: 执行查询 (rag_query)
        query = "What is the transformer architecture?"
        print(f"\n[2] Querying Knowledge Base: '{query}'...")
        query_result = await client.call_tool("query_knowledge_base", {
            "worker_id": worker_id, 
            "query": query,
            "top_k": 2
        })
        print(f"Result:\n{query_result}")
        
        # 5. 调用工具: 释放资源
        print(f"\n[3] Releasing Session...")
        await client.call_tool("release_rag_session", {"worker_id": worker_id})
        print("✅ Session Released")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(run_test())