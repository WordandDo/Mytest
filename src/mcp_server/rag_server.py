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
        
        # 将top_k存储到session中供后续查询使用
        session["config_top_k"] = top_k
        
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
    
    此函数已移除客户端探活机制，通过设置长超时来支持服务端排队。
    """
    req_timeout = 600.0  # 设置600秒的超时，允许在服务端排队
    target_resource_type = "rag"

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                f"{RESOURCE_API_URL}/allocate",
                json={
                    "worker_id": worker_id,
                    "type": target_resource_type,
                    "timeout": req_timeout
                },
                timeout=req_timeout + 5  # 客户端超时略长于逻辑超时
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.TimeoutException:
            return json.dumps({
                "status": "error",
                "message": f"System busy: Could not acquire RAG resource within {req_timeout}s. Please try again later."
            })
        except Exception as e:
            return json.dumps({"status": "error", "message": f"RAG allocation failed: {str(e)}"})

    resource_id = data.get("id")
    token = data.get("token")
    RAG_SESSIONS[worker_id] = {"resource_id": resource_id, "token": token}
    
    return json.dumps({
        "status": "success",
        "message": "Connected",
        "resource_id": resource_id
    })

@ToolRegistry.register_tool("rag_query")
async def query_knowledge_base(worker_id: str, query: str, top_k: Optional[int] = None) -> str:
    """远程查询知识库"""
    session = RAG_SESSIONS.get(worker_id)
    if not session:
        return json.dumps({"status": "error", "message": "No active RAG session. Call setup_rag_engine first."})

    if not query:
        return json.dumps({"status": "error", "message": "Query cannot be empty"})

    # 【关键逻辑调整】
    # 1. 如果 Agent 传了值，用 Agent 的
    # 2. 如果 Agent 没传，优先用 Task 初始化时注入的配置 (session["config_top_k"])
    # 3. 如果 Task 也没配置，传 None 给后端，让后端使用 deployment_config.json 中的 default_top_k
    
    # 尝试从 Session 获取 Task 级配置 (前提是你实现了上一轮建议的 rag_initialization 修改)
    task_config_top_k = session.get("config_top_k") if session else None
    
    # 决策最终的 effective_top_k
    if top_k is not None:
        effective_top_k = top_k          # Agent 显式指定，优先级最高
    elif task_config_top_k is not None:
        effective_top_k = task_config_top_k # Task 配置，优先级次之
    else:
        effective_top_k = None           # 传 None，触发后端读取 deployment_config.json

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{RESOURCE_API_URL}/query_rag",
                json={
                    "resource_id": session["resource_id"],
                    "worker_id": worker_id,
                    "query": query,
                    "top_k": effective_top_k  # 发送 None 或具体数值
                },
                timeout=120
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