# src/mcp_server/rag_server.py
import sys
import os
import json
import httpx
from typing import Optional, Dict
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# 环境设置
load_dotenv()
cwd = os.getcwd()
sys.path.append(cwd)
if os.path.join(cwd, "src") not in sys.path:
    sys.path.append(os.path.join(cwd, "src"))

# 导入真正的索引加载器
from utils.rag_index import get_rag_index_class, BaseRAGIndex

from mcp_server.core.probe import wait_for_resource_availability
from mcp_server.core.registry import ToolRegistry

mcp = FastMCP("RAG Specialized Gateway")
RESOURCE_API_URL = os.environ.get("RESOURCE_API_URL", "http://localhost:8000")

# 全局会话，存储 worker_id -> 索引实例的映射
# 结构: { worker_id: { "resource_id": str, "index": BaseRAGIndex, "path": str } }
RAG_SESSIONS: Dict[str, Dict] = {}

print("🚀 Starting RAG MCP Server")

@ToolRegistry.register_tool("rag_lifecycle")
@mcp.tool()
async def setup_rag_engine(worker_id: str) -> str:
    """
    初始化 RAG 引擎：向资源管理器申请 RAG 资源并加载索引到内存。
    这可能需要几秒钟时间来加载模型和向量数据。
    """
    
    # 1. 资源探活
    print(f"[{worker_id}] Probing RAG availability...")
    is_available = await wait_for_resource_availability(
        api_url=RESOURCE_API_URL,
        resource_type="rag",
        timeout=60 # RAG 释放可能较快，多给点等待时间
    )
    
    if not is_available:
        return json.dumps({
            "status": "error", 
            "message": "System busy: No RAG slots available."
        })

    print(f"[{worker_id}] Requesting RAG resource...")
    async with httpx.AsyncClient() as client:
        try:
            # 2. 申请资源 (type="rag")
            resp = await client.post(
                f"{RESOURCE_API_URL}/allocate", 
                json={"worker_id": worker_id, "type": "rag"}, 
                timeout=60
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            error_msg = f"RAG Alloc failed: {e}"
            print(error_msg)
            return json.dumps({"status": "error", "message": error_msg})

    resource_id = data.get("id")
    index_path = data.get("index_path")
    model_name = data.get("emb_model")
    use_faiss = data.get("use_faiss", False)

    print(f"[{worker_id}] Allocated {resource_id}. Loading index from {index_path}...")

    try:
        # 获取对应的索引类并加载
        IndexClass = get_rag_index_class(use_faiss=use_faiss)
        
        # 这里的 device 可以根据部署情况调整，默认为 cpu 以节省显存给主 Agent
        loaded_index = IndexClass.load_index(
            index_path=index_path,
            model_name=model_name,
            device="cpu" 
        )
        
        RAG_SESSIONS[worker_id] = {
            "resource_id": resource_id,
            "index": loaded_index,
            "index_path": index_path,
            "status": "active"
        }
        
        msg = f"RAG Engine ready. Loaded {len(loaded_index.chunks)} chunks."
        print(f"[{worker_id}] {msg}")
        
        return json.dumps({
            "status": "success",
            "message": msg,
            "resource_id": resource_id
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return json.dumps({"status": "error", "message": f"Failed to load local index: {str(e)}"})

@ToolRegistry.register_tool("rag_query")
@mcp.tool()
async def query_knowledge_base(worker_id: str, query: str, top_k: int = 3) -> str:
    """
    查询知识库。根据语义相似度检索相关上下文。
    必须先调用 setup_rag_engine 初始化。
    """
    session = RAG_SESSIONS.get(worker_id)
    if not session:
        return json.dumps({"status": "error", "message": "No active RAG session. Call setup_rag_engine first."})

    rag_index = session.get("index")
    if not rag_index:
        return json.dumps({"status": "error", "message": "RAG index not loaded properly."})
    
    if not query:
        return json.dumps({"status": "error", "message": "Query cannot be empty"})

    try:
        # 调用真正的查询接口
        # query 方法返回的是格式化后的字符串 "### Retrieved Context:\n..."
        result_text = rag_index.query(query, top_k=top_k)
        
        return json.dumps({
            "status": "success",
            "results": result_text
        })

    except Exception as e:
        return json.dumps({"status": "error", "message": f"Query execution failed: {str(e)}"})

@ToolRegistry.register_tool("rag_lifecycle")
@mcp.tool()
async def release_rag_engine(worker_id: str) -> str:
    """释放 RAG 资源并卸载内存中的索引"""
    session = RAG_SESSIONS.get(worker_id)
    if session:
        resource_id = session.get("resource_id")
        print(f"[{worker_id}] Releasing resource {resource_id}...")
        
        # 1. 释放远程资源
        async with httpx.AsyncClient() as client:
            try:
                await client.post(
                    f"{RESOURCE_API_URL}/release", 
                    json={"resource_id": resource_id, "worker_id": worker_id}, 
                    timeout=10
                )
            except Exception as e:
                print(f"Warning: Failed to notify resource manager: {e}")
        
        # 2. 清理本地内存
        # 显式删除索引对象以辅助 GC
        if "index" in session:
            del session["index"]
        RAG_SESSIONS.pop(worker_id, None)
        
        import gc
        gc.collect()
        
    return "Released"

if __name__ == "__main__":
    # RAG Server 运行在 8081 端口
    mcp.settings.debug = True
    mcp.settings.host = "0.0.0.0"
    mcp.settings.port = 8081
    
    print(f"🚀 Starting RAG MCP Server on {mcp.settings.host}:{mcp.settings.port} (SSE Mode)...")
    mcp.run(transport='sse')