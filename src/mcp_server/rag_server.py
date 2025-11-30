# src/mcp_server/rag_server.py
import sys
import os
import json
import httpx

# 动态添加 src 路径，确保能找到 mcp_server 模块
cwd = os.getcwd()
if os.path.join(cwd, "src") not in sys.path:
    sys.path.append(os.path.join(cwd, "src"))

from typing import Optional, Dict
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from mcp_server.core.probe import wait_for_resource_availability
from mcp_server.core.registry import ToolRegistry

# 环境设置
load_dotenv()
mcp = FastMCP("RAG Specialized Gateway")
RESOURCE_API_URL = os.environ.get("RESOURCE_API_URL", "http://localhost:8000")

# 全局会话，存储 worker_id -> 令牌信息
# 结构: { worker_id: { "resource_id": str, "token": str } }
RAG_SESSIONS: Dict[str, Dict] = {}

print("🚀 Starting RAG MCP Server (Client Mode)")

@ToolRegistry.register_tool("rag_lifecycle")
@mcp.tool()
async def setup_rag_session(worker_id: str) -> str:
    """初始化 RAG 会话：申请访问 Token。
    (原名 setup_rag_engine)
    """
    is_available = await wait_for_resource_availability(
        RESOURCE_API_URL, "rag", timeout=60
    )
    if not is_available:
        return json.dumps({"status": "error", "message": "System busy: No RAG slots available."})

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                f"{RESOURCE_API_URL}/allocate",
                json={"worker_id": worker_id, "type": "rag"},
                timeout=60
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    resource_id = data.get("id")
    token = data.get("token")
    RAG_SESSIONS[worker_id] = {"resource_id": resource_id, "token": token}
    
    return json.dumps({
        "status": "success",
        "message": "Connected",
        "resource_id": resource_id
    })

@ToolRegistry.register_tool("rag_query")
@mcp.tool()
async def query_knowledge_base(worker_id: str, query: str, top_k: int = 3) -> str:
    """远程查询知识库"""
    session = RAG_SESSIONS.get(worker_id)
    if not session:
        return json.dumps({"status": "error", "message": "No active RAG session. Call setup_rag_engine first."})

    if not query:
        return json.dumps({"status": "error", "message": "Query cannot be empty"})

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{RESOURCE_API_URL}/query_rag",
                json={
                    "resource_id": session["resource_id"],
                    "worker_id": worker_id,
                    "query": query,
                    "top_k": top_k
                },
                timeout=30
            )
            if resp.status_code != 200:
                return json.dumps({"status": "error", "message": f"Remote Error: {resp.text}"})
            return json.dumps({"status": "success", "results": resp.json().get("results", "")})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})

@ToolRegistry.register_tool("rag_lifecycle")
@mcp.tool()
async def release_rag_session(worker_id: str) -> str:
    """释放 RAG 资源会话"""
    session = RAG_SESSIONS.pop(worker_id, None)
    if session:
        async with httpx.AsyncClient() as client:
            try:
                await client.post(
                    f"{RESOURCE_API_URL}/release",
                    json={"resource_id": session["resource_id"], "worker_id": worker_id}
                )
            except Exception as e:
                pass
    return "Released"

if __name__ == "__main__":
    mcp.run()