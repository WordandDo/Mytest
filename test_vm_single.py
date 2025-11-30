import asyncio
import sys
import os
import uuid
import json

# 添加 src 到路径以加载 utils
sys.path.append(os.path.join(os.getcwd(), "src"))

from utils.mcp_client import MCPClientWrapper

async def run_vm_test():
    # 1. 指定 VM 对应的 MCP Server 脚本
    server_script = "src/mcp_server/osworld_server.py"
    
    print(f"📡 Connecting to MCP Server: {server_script}...")
    
    # 2. 启动客户端 (stdio 模式连接子进程)
    client = MCPClientWrapper(server_script)
    
    try:
        await client.connect()
        print("✅ MCP Connection Established")
        
        # 生成测试用的 worker_id
        worker_id = f"test-vm-{uuid.uuid4().hex[:4]}"
        
        # 3. 调用工具: 申请 VM 会话 (setup_vm_session)
        # 对应 @ToolRegistry.register_tool("computer_lifecycle")
        print(f"\n[1] Allocating VM Session for {worker_id}...")
        print("   (Note: This requests a REAL VM from Resource API. Expect failure if no credentials configured.)")
        
        init_result = await client.call_tool("setup_vm_session", {
            "config_name": "default",
            "task_id": "test_task_001",
            "worker_id": worker_id
        })
        print(f"Result: {init_result}")
        
        # 检查是否成功
        try:
            res_json = json.loads(init_result)
        except:
            res_json = {"status": "error", "message": str(init_result)}

        if res_json.get("status") == "error":
            print("\n❌ Setup failed (Expected if VM credentials are missing).")
            print(f"Reason: {res_json.get('message')}")
            return

        # 4. 如果申请成功，尝试获取屏幕截图 (get_observation)
        print(f"\n[2] Getting Desktop Observation...")
        obs_result = await client.call_tool("get_observation", {"worker_id": worker_id})
        print(f"Result (truncated): {obs_result[:100]}...")

        # 5. 尝试移动鼠标 (desktop_mouse_move)
        print(f"\n[3] Moving Mouse...")
        move_result = await client.call_tool("desktop_mouse_move", {
            "worker_id": worker_id, 
            "x": 500, 
            "y": 500
        })
        print(f"Result: {move_result}")

        # 6. 释放资源
        print(f"\n[4] Releasing Session...")
        await client.call_tool("teardown_environment", {"worker_id": worker_id})
        print("✅ Session Released")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(run_vm_test())