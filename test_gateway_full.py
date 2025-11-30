import asyncio
import json
from mcp.client.sse import sse_client
from mcp import ClientSession

# 网关地址
GATEWAY_URL = "http://localhost:8080/sse"

async def test_full_gateway():
    print(f"🔌 Connecting to Gateway at {GATEWAY_URL}...")
    
    try:
        # 建立 SSE 连接
        async with sse_client(GATEWAY_URL) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                print("✅ Gateway Connected!")
                
                # 1. 列出所有可用工具
                print("\n📋 Listing Available Tools...")
                tools = await session.list_tools()
                tool_names = [t.name for t in tools.tools]
                print(f"   Found {len(tool_names)} tools: {tool_names}")
                
                # 检查关键工具是否存在
                if "query_knowledge_base" not in tool_names or "desktop_mouse_click" not in tool_names:
                    print("❌ Critical tools missing! Check Gateway logs.")
                    return

                # --- 测试 RAG 模块 ---
                print("\n🧠 [Test 1] Testing RAG Module...")
                # 步骤 A: 初始化 RAG
                worker_id = "test-user-001"
                print(f"   -> Allocating RAG session for {worker_id}...")
                await session.call_tool("setup_rag_session", {"worker_id": worker_id})
                
                # 步骤 B: 查询
                query = "What is the capital of France?" # 简单问题测试
                print(f"   -> Querying: {query}")
                rag_res = await session.call_tool("query_knowledge_base", {
                    "worker_id": worker_id, 
                    "query": query
                })
                # 解析并打印结果（截取前100字符）
                res_text = rag_res.content[0].text if rag_res.content else "No response"
                print(f"   -> Result: {res_text[:100]}...")
                
                # --- 测试 VM 模块 ---
                print("\n🖥️  [Test 2] Testing VM Module (Real Aliyun ECS)...")
                # 步骤 A: 申请 VM (这会调用后端去锁定一台空闲的阿里云机器)
                print(f"   -> Allocating VM session...")
                vm_res = await session.call_tool("setup_vm_session", {
                    "config_name": "default",
                    "task_id": "integration_test",
                    "worker_id": worker_id
                })
                
                # 解析返回结果
                vm_data = json.loads(vm_res.content[0].text)
                if vm_data.get("status") == "success":
                    print("   ✅ VM Allocated Successfully!")
                    # 步骤 B: 移动鼠标
                    print("   -> Moving mouse to (500, 500)...")
                    await session.call_tool("desktop_mouse_move", {
                        "worker_id": worker_id,
                        "x": 500,
                        "y": 500
                    })
                    print("   ✅ Action Executed")
                    
                    # 步骤 C: 释放环境
                    print("   -> Teardown VM environment...")
                    await session.call_tool("teardown_environment", {"worker_id": worker_id})
                else:
                    print(f"   ❌ VM Allocation Failed: {vm_data.get('message')}")

                # 释放 RAG
                await session.call_tool("release_rag_session", {"worker_id": worker_id})
                print("\n✅ All Tests Completed!")

    except Exception as e:
        print(f"\n❌ Connection Error: {e}")
        print("Hint: Make sure 'bash start_gateway.sh' is running in another terminal.")

if __name__ == "__main__":
    asyncio.run(test_full_gateway())