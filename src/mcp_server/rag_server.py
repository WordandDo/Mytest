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

print("🚀 Starting RAG MCP Server (Client Mode)")

# RAG初始化函数
async def rag_initialization(worker_id: str, config_content: str = "") -> bool:
    """
    RAG资源初始化函数，用于解析Benchmark特有的数据结构并执行初始化操作
    
    Args:
        worker_id: 工作进程ID
        config_content: 初始化配置内容，应该是包含knowledge_base_ids和top_k的JSON
        
    Returns:
        bool: 初始化是否成功
    """
    # 防御性编程：无配置即成功
    if not config_content:
        return True
    
    try:
        # 解析配置
        config = json.loads(config_content) if isinstance(config_content, str) else config_content
        
        # 获取知识库ID列表和top_k参数
        knowledge_base_ids = config.get("knowledge_base_ids", [])
        top_k = config.get("top_k", 3)
        
        # 如果没有知识库需要设置，直接返回成功
        if not knowledge_base_ids:
            return True
        
        # 调用set_rag_context工具更新RAG上下文
        # 这里应该调用相应的工具来设置知识库上下文
        # 由于当前代码中没有看到set_rag_context工具，我们需要假设它存在或稍后实现
        session = RAG_SESSIONS.get(worker_id)
        if not session:
            raise RuntimeError(f"No active RAG session for worker: {worker_id}")
        
        # TODO: 实际调用设置知识库上下文的工具
        # 这可能需要与Resource API通信或直接调用相应的函数
        
        return True
    except Exception as e:
        print(f"RAG initialization failed for worker {worker_id}: {e}")
        return False

# 全局会话，存储 worker_id -> 令牌信息
# 结构: { worker_id: { "resource_id": str, "token": str } }
RAG_SESSIONS: Dict[str, Dict] = {}

@ToolRegistry.register_tool("rag_lifecycle")
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