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

from typing import Optional, Dict, Any
from dotenv import load_dotenv

from mcp_server.core.registry import ToolRegistry

# 设置日志
logger = logging.getLogger("RAGServer")

# 环境设置
load_dotenv()
RESOURCE_API_URL = os.environ.get("RESOURCE_API_URL", "http://localhost:8000")

# 全局会话字典，Key 为 worker_id
# 结构: { worker_id: { "resource_id": str, "token": str, "base_url": str, "config_top_k": int } }
RAG_SESSIONS: Dict[str, Dict[str, Any]] = {}

# =============================================================================
# RAG Initialization Function (Batch Allocation Pattern)
# =============================================================================

async def rag_initialization(worker_id: str, config_content = None) -> bool:
    """
    RAG resource initialization function, automatically called by setup_batch_resources.

    This function is invoked after RAG resources have been allocated and synced to RAG_SESSIONS.
    It handles any additional initialization logic, such as parsing and storing configuration
    parameters (e.g., top_k) from the task config.

    Args:
        worker_id: Unique worker identifier
        config_content: Optional configuration content (JSON string or dict) containing
                       initialization parameters like top_k

    Returns:
        bool: True if initialization succeeded, False otherwise
    """
    session = RAG_SESSIONS.get(worker_id)
    if not session:
        logger.error(f"[{worker_id}] RAG session not found in RAG_SESSIONS. "
                    "Ensure _sync_resource_sessions was called first.")
        return False

    # If configuration is provided, parse and store parameters
    if config_content:
        try:
            config_dict = json.loads(config_content) if isinstance(config_content, str) else config_content

            # Store task-level top_k configuration
            if "top_k" in config_dict:
                session["config_top_k"] = config_dict["top_k"]
                logger.info(f"[{worker_id}] RAG initialization completed with top_k={config_dict['top_k']}")
            else:
                logger.info(f"[{worker_id}] RAG initialization completed (no top_k in config)")
        except Exception as e:
            logger.error(f"[{worker_id}] RAG initialization failed to parse config: {e}")
            return False
    else:
        logger.info(f"[{worker_id}] RAG initialization completed (no config provided)")

    return True


# =============================================================================
# RAG Lifecycle Tools (Self-Managed Pattern - DEPRECATED)
# =============================================================================
# NOTE: setup_rag_session is now DEPRECATED in favor of the unified batch allocation pattern.
# It is kept for backward compatibility but should not be used in new code.

@ToolRegistry.register_tool("rag_lifecycle")
async def setup_rag_session(worker_id: str, config: str = "{}") -> str:
    """
    [HIDDEN] [DEPRECATED] Legacy setup function - use unified batch allocation instead.

    This function is kept for backward compatibility only. New code should use
    the unified batch allocation pattern via allocate_batch_resources.

    For legacy callers, this function now simply checks if a session was already
    established via batch allocation and applies any additional configuration.

    Args:
        worker_id: Unique worker identifier
        config: Optional JSON configuration string containing initialization parameters (e.g., top_k)

    Returns:
        JSON string with status and connection details
    """
    logger.warning(f"[{worker_id}] setup_rag_session is DEPRECATED. Use batch allocation instead.")

    # Check if session already exists (from batch allocation)
    if worker_id in RAG_SESSIONS:
        session = RAG_SESSIONS[worker_id]

        # Parse and apply configuration if provided
        try:
            config_dict = json.loads(config) if isinstance(config, str) else config
            if config_dict.get("top_k"):
                session["config_top_k"] = config_dict["top_k"]
                logger.info(f"[{worker_id}] Updated existing session with config: {config_dict}")
        except Exception as e:
            logger.error(f"[{worker_id}] Failed to parse config: {e}")

        return json.dumps({
            "status": "success",
            "message": "Using existing session from batch allocation",
            "resource_id": session.get("resource_id"),
            "base_url": session.get("base_url")
        })

    # If no session exists, this is an error - batch allocation should have been called first
    error_msg = ("No RAG session found. This function is deprecated. "
                "Use allocate_batch_resources(['rag']) via the parent environment instead.")
    logger.error(f"[{worker_id}] {error_msg}")
    return json.dumps({
        "status": "error",
        "message": error_msg
    })

@ToolRegistry.register_tool("rag_query")
async def query_knowledge_base(worker_id: str, query: str, top_k: Optional[int] = None) -> str:
    """
    Query the knowledge base using RAG (Direct Connection).

    This tool performs semantic search on the allocated knowledge base resources.
    Top-k parameter follows priority: Agent explicit value > Task config > Backend default.

    Args:
        worker_id: Unique worker identifier
        query: The query string to search for
        top_k: Optional number of top results to return (overrides task config)

    Returns:
        JSON string with query results
    """
    # Check if session exists
    session = RAG_SESSIONS.get(worker_id)
    if not session:
        error_msg = "No active RAG session. Call setup_rag_session first."
        logger.error(f"[{worker_id}] {error_msg}")
        return json.dumps({"status": "error", "message": error_msg})

    if not query:
        return json.dumps({"status": "error", "message": "Query cannot be empty"})

    # Determine effective top_k with priority: Agent > Task Config > Backend Default
    # 1. If Agent explicitly passed a value, use it (highest priority)
    # 2. If not, use Task config top_k stored in session (from setup_rag_session)
    # 3. If neither exists, pass None to backend, triggering backend's default_top_k from deployment_config.json

    task_config_top_k = session.get("config_top_k") if session else None

    if top_k is not None:
        effective_top_k = top_k  # Agent explicit specification (highest priority)
    elif task_config_top_k is not None:
        effective_top_k = task_config_top_k  # Task configuration (second priority)
    else:
        effective_top_k = None  # Pass None to trigger backend default

    target_url = session["base_url"]  # Direct connection URL

    try:
        async with httpx.AsyncClient() as client:
            logger.info(f"[{worker_id}] Querying RAG service at {target_url}/query with top_k={effective_top_k}...")
            # POST directly to RAG Service
            resp = await client.post(
                f"{target_url}/query",
                json={
                    "query": query,
                    "top_k": effective_top_k if effective_top_k else 5,  # Ensure integer is passed to backend
                    "token": session.get("token", "")
                },
                timeout=120
            )

            if resp.status_code != 200:
                error_msg = f"RAG Service Error: {resp.text}"
                logger.error(f"[{worker_id}] {error_msg}")
                return json.dumps({"status": "error", "message": error_msg})

            # Return results directly
            results = resp.json().get("results", "")
            logger.info(f"[{worker_id}] Query completed successfully.")
            return json.dumps({"status": "success", "results": results})

    except httpx.TimeoutException:
        error_msg = "Query timeout: RAG service did not respond in time"
        logger.error(f"[{worker_id}] {error_msg}")
        return json.dumps({"status": "error", "message": error_msg})
    except Exception as e:
        error_msg = f"Direct connection failed: {str(e)}"
        logger.error(f"[{worker_id}] {error_msg}")
        return json.dumps({"status": "error", "message": error_msg})

@ToolRegistry.register_tool("rag_lifecycle")
async def teardown_rag_session(worker_id: str) -> str:
    """
    [HIDDEN] Teardown RAG session and release resources.

    This tool releases the allocated RAG resources back to the resource pool
    and removes the local session registration.

    Args:
        worker_id: Unique worker identifier

    Returns:
        Status message string
    """
    session = RAG_SESSIONS.pop(worker_id, None)

    if not session:
        logger.warning(f"[{worker_id}] No active RAG session to release.")
        return "No active session"

    resource_id = session.get("resource_id")

    # Call backend API to release resource
    async with httpx.AsyncClient() as client:
        try:
            logger.info(f"[{worker_id}] Releasing RAG resource {resource_id}...")
            await client.post(
                f"{RESOURCE_API_URL}/release",
                json={"resource_id": resource_id, "worker_id": worker_id},
                timeout=10
            )
            logger.info(f"[{worker_id}] RAG resource released successfully.")
        except Exception as e:
            logger.error(f"[{worker_id}] Failed to release RAG resource: {e}")
            # Continue even if release fails to ensure local cleanup

    return "Released"