# src/mcp_server/rag_server.py
import sys
import os
import json
import httpx
import logging

# 动态添加 src 路径，确保能找到 mcp_server 模块
cwd = os.getcwd()
if os.path.join(cwd, "src") not in sys.path:
    sys.path.append(os.path.join(cwd, "src"))

from typing import Optional, Dict, Any, Annotated
from dotenv import load_dotenv
from pydantic import Field

from mcp_server.core.registry import ToolRegistry

# 设置日志
logger = logging.getLogger("RAGServer")

# 环境设置
load_dotenv()
RESOURCE_API_URL = os.environ.get("RESOURCE_API_URL", "http://localhost:8000")

# 全局会话字典
GLOBAL_SESSIONS: Dict[str, Dict[str, Any]] = {}

# =============================================================================
# 1. 核心共享逻辑 (Shared Core Logic)
# =============================================================================

async def _safe_release_resource(resource_id: str, worker_id: str, reason: str = ""):
    """
    尝试释放单个 RAG 资源的保护逻辑，供 teardown 或回滚使用。
    """
    if not resource_id:
        return
    async with httpx.AsyncClient() as client:
        try:
            await client.post(
                f"{RESOURCE_API_URL}/release",
                json={"resource_id": resource_id, "worker_id": worker_id},
                timeout=10
            )
        except Exception as e:
            logger.error(f"[{worker_id}] Safe release failed for RAG resource {resource_id} ({reason}): {e}")

async def _configure_rag_session(worker_id: str, config_data: Any) -> bool:
    """
    [Core Logic] 统一的 RAG 会话配置逻辑。
    负责解析配置参数（如 top_k）并应用到 Session 中。
    """
    session = GLOBAL_SESSIONS.get(worker_id)
    if not session:
        logger.error(f"[{worker_id}] Session not found during configuration.")
        return False

    if not config_data:
        logger.info(f"[{worker_id}] No config provided, keeping default state.")
        return True

    try:
        # 支持 JSON 字符串或 Dict
        config_dict = config_data
        if isinstance(config_data, str):
            config_dict = json.loads(config_data)
            
        # 应用配置
        if "top_k" in config_dict:
            session["config_top_k"] = config_dict["top_k"]
            logger.info(f"[{worker_id}] Configured top_k={config_dict['top_k']}")
            
        return True
    except Exception as e:
        logger.error(f"[{worker_id}] Failed to parse RAG config: {e}")
        return False

async def _cleanup_rag_session_local(worker_id: str):
    """
    [Core Logic] 统一的本地状态清理逻辑。
    """
    if worker_id in GLOBAL_SESSIONS:
        del GLOBAL_SESSIONS[worker_id]
        logger.info(f"[{worker_id}] RAG Session local state cleaned up.")

# =============================================================================
# 2. 批处理钩子 (Batch Hooks) - 供 system_tools 调用
# =============================================================================

async def rag_initialization(worker_id: str, config_content = None) -> bool:
    """
    [Setup Hook] 批处理初始化钩子。
    """
    return await _configure_rag_session(worker_id, config_content)

async def rag_cleanup(worker_id: str):
    """
    [Teardown Hook] 批处理清理钩子。
    """
    await _cleanup_rag_session_local(worker_id)

# =============================================================================
# 3. 独立工具 (Standalone Tools) - 供 MCP Client 直接调用
# =============================================================================

@ToolRegistry.register_tool("rag_lifecycle")
async def setup_rag_session(
    worker_id: Annotated[str, Field(description="Worker id/session key (auto-injected by HttpMCPEnv).")],
    config: Annotated[str, Field(description="JSON string for RAG config, e.g., {'top_k': 5}.", min_length=1)] = "{}"
) -> str:
    """
    [Legacy Setup] 更新当前 RAG Session 的配置（top_k 等）。
    通常资源由 system_tools 分配，此工具仅调整参数。
    worker_id 由客户端自动注入。
    """
    if worker_id not in GLOBAL_SESSIONS:
        return json.dumps({
            "status": "error", 
            "message": "RAG session must be allocated via system_tools first."
        })
    
    await _configure_rag_session(worker_id, config)
    
    session = GLOBAL_SESSIONS[worker_id]
    return json.dumps({
        "status": "success",
        "resource_id": session.get("resource_id"),
        "base_url": session.get("base_url")
    })

@ToolRegistry.register_tool("rag_lifecycle")
async def teardown_rag_session(
    worker_id: Annotated[str, Field(description="Worker id/session key (auto-injected by HttpMCPEnv).")]
) -> str:
    """
    [Standalone Teardown] 主动释放 RAG 资源并清理会话。
    """
    # 1. 资源层释放
    session = GLOBAL_SESSIONS.get(worker_id)
    if session:
        await _safe_release_resource(session.get("resource_id"), worker_id, reason="teardown_rag_session")

    # 2. 会话层清理
    await _cleanup_rag_session_local(worker_id)
    return "Released"

# =============================================================================
# 4. 查询工具 (Query Tools)
# =============================================================================

@ToolRegistry.register_tool("rag_query")
async def query_knowledge_base_dense(
    worker_id: Annotated[str, Field(description="Worker id/session key (auto-injected by HttpMCPEnv).")],
    query: Annotated[str, Field(description="Natural language question or statement to retrieve against the KB.", min_length=1)],
    top_k: Annotated[Optional[int], Field(description="Override for number of results; positive integer.", ge=1)] = None
) -> str:
    """[Dense Search] Semantic vector search over the knowledge base. Falls back to configured top_k or 5."""
    return await _internal_query(worker_id, query, top_k, search_type="dense")

@ToolRegistry.register_tool("rag_query_sparse")
async def query_knowledge_base_sparse(
    worker_id: Annotated[str, Field(description="Worker id/session key (auto-injected by HttpMCPEnv).")],
    query: Annotated[str, Field(description="Keyword-friendly query text; include specific terms/IDs.", min_length=1)],
    top_k: Annotated[Optional[int], Field(description="Override for number of results; positive integer.", ge=1)] = None
) -> str:
    """[Sparse Search] Keyword/BM25 style search over the knowledge base. Falls back to configured top_k or 5."""
    return await _internal_query(worker_id, query, top_k, search_type="sparse")

async def _internal_query(worker_id: str, query: str, top_k: Optional[int], search_type: str) -> str:
    session = GLOBAL_SESSIONS.get(worker_id)
    if not session:
        return json.dumps({"status": "error", "message": "No active RAG session."})

    task_config_top_k = session.get("config_top_k")
    effective_top_k = top_k if top_k is not None else task_config_top_k
    target_url = session["base_url"]

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{target_url}/query",
                json={
                    "query": query,
                    "top_k": effective_top_k if effective_top_k else 5,
                    "token": session.get("token", ""),
                    "search_type": search_type
                },
                timeout=120
            )
            if resp.status_code != 200:
                return json.dumps({"status": "error", "message": f"RAG Service Error: {resp.text}"})
            
            results = resp.json().get("results", "")
            return json.dumps({"status": "success", "results": results})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})
