# src/mcp_server/main.py
from src.mcp_server.osworld_server import mcp

if __name__ == "__main__":
    import uvicorn
    # 启动 HTTP/SSE 模式
    # FastMCP 默认会挂载 endpoints，例如 /sse
    mcp.settings.debug = True
    uvicorn.run(mcp._fastapi_app, host="0.0.0.0", port=8080)