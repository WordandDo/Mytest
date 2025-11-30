import httpx
import asyncio

async def check_gateway_tools():
    gateway_url = "http://localhost:8080/sse"
    print(f"🔍 Checking Gateway at {gateway_url}...")
    
    # SSE 握手通常比较复杂，这里我们简单检查端口是否存活
    # 真正的 MCP 交互需要完整的 SSE 客户端实现
    try:
        async with httpx.AsyncClient() as client:
            # 尝试访问根路径或其他端点看服务是否响应
            resp = await client.get("http://localhost:8080/messages", params={"session_id": "test"})
            # 404 或 405 也是正常的，说明服务在运行但需要正确的 SSE 请求
            if resp.status_code in [200, 404, 405, 422]:
                print(f"✅ Gateway is running (Status: {resp.status_code})")
                print("You can now configure your MCP client to connect to http://0.0.0.0:8080/sse")
            else:
                print(f"⚠️ Unexpected status: {resp.status_code}")
    except Exception as e:
        print(f"❌ Could not connect to Gateway: {e}")
        print("Make sure you ran 'bash start_gateway.sh'")

if __name__ == "__main__":
    asyncio.run(check_gateway_tools())