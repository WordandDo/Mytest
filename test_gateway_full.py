import asyncio
import json
import base64  # [新增] 用于解码图片
import os      # [新增] 用于路径处理
from mcp.client.sse import sse_client
from mcp import ClientSession

# 网关地址
GATEWAY_URL = "http://localhost:8080/sse"

# [新增] 辅助函数：保存 Base64 图片
def save_screenshot(b64_str, filename):
    if not b64_str:
        print(f"   ⚠️ No screenshot data for {filename}")
        return
    try:
        with open(filename, "wb") as f:
            f.write(base64.b64decode(b64_str))
        print(f"   🖼️ Saved screenshot to {filename}")
    except Exception as e:
        print(f"   ❌ Failed to save screenshot: {e}")

async def test_full_gateway():
    print(f"🔌 Connecting to Gateway at {GATEWAY_URL}...")
    
    # 确保输出目录存在
    os.makedirs("test_output", exist_ok=True)

    try:
        # 建立 SSE 连接
        async with sse_client(GATEWAY_URL) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                print("✅ Gateway Connected!")
                
                # --- 测试 VM 模块 ---
                print("\n🖥️  [Test 2] Testing VM Module (with Recording & Screenshots)...")
                worker_id = "test-user-001"
                
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
                    
                    # [修改点 1] 保存初始截图
                    init_shot = vm_data.get("observation", {}).get("screenshot")
                    save_screenshot(init_shot, "test_output/01_vm_init.png")

                    # [修改点 2] 开始录制视频
                    print("   🎥 Starting recording...")
                    await session.call_tool("start_recording", {"worker_id": worker_id})
                    
                    # 步骤 B: 移动鼠标
                    print("   -> Moving mouse to (500, 500)...")
                    await session.call_tool("desktop_mouse_move", {
                        "worker_id": worker_id,
                        "x": 500,
                        "y": 500
                    })
                    print("   ✅ Action Executed")
                    
                    # [修改点 3] 主动获取操作后的截图
                    print("   -> Fetching observation after move...")
                    obs_res = await session.call_tool("get_observation", {"worker_id": worker_id})
                    obs_data = json.loads(obs_res.content[0].text)
                    save_screenshot(obs_data.get("screenshot"), "test_output/02_after_move.png")

                    # [修改点 4] 停止录制并保存
                    # 注意：路径是服务器端的绝对路径
                    video_path = os.path.abspath("test_output/session_video.mp4")
                    print(f"   ⏹️ Stopping recording (saving to server: {video_path})...")
                    await session.call_tool("stop_recording", {
                        "worker_id": worker_id, 
                        "save_path": video_path
                    })

                    # 步骤 C: 释放环境
                    print("   -> Teardown VM environment...")
                    await session.call_tool("teardown_environment", {"worker_id": worker_id})
                else:
                    print(f"   ❌ VM Allocation Failed: {vm_data.get('message')}")

                print("\n✅ All Tests Completed!")

    except Exception as e:
        print(f"\n❌ Connection Error: {e}")
        print("Hint: Make sure 'bash start_gateway.sh' is running in another terminal.")

if __name__ == "__main__":
    asyncio.run(test_full_gateway())