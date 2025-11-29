# src/mcp_server/rag_server.py
import sys
import os
import json
import httpx
import asyncio
from typing import Optional, List
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# 环境设置
load_dotenv()
cwd = os.getcwd()
sys.path.append(cwd)

mcp = FastMCP("RAG Specialized Gateway")
RESOURCE_API_URL = os.environ.get("RESOURCE_API_URL", "http://localhost:8000")

# 全局会话，存储 RAG 资源信息
RAG_SESSIONS = {}

print("🚀 Starting RAG MCP Server")

@mcp.tool()
async def setup_rag_engine(worker_id: str) -> str:
    """
    初始化 RAG 引擎：向资源管理器申请 RAG 资源。
    """
    async with httpx.AsyncClient() as client:
        try:
            # 申请 rag 类型的资源
            resp = await client.post(
                f"{RESOURCE_API_URL}/allocate", 
                json={"worker_id": worker_id, "type": "rag"}, 
                timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return json.dumps({"status": "error", "message": f"RAG Alloc failed: {e}"})

    resource_id = data.get("id")
    index_path = data.get("index_path")

    # 在这里可以真正加载索引或初始化查询对象
    # 为了演示，我们将配置存入会话
    RAG_SESSIONS[worker_id] = {
        "resource_id": resource_id,
        "index_path": index_path,
        "status": "active"
    }
    
    return json.dumps({
        "status": "success",
        "message": f"RAG Engine ready. Index: {index_path}",
        "resource_id": resource_id
    })

@mcp.tool()
async def query_knowledge_base(worker_id: str, query: str, top_k: int = 3) -> str:
    """
    查询知识库。必须先调用 setup_rag_engine。
    """
    session = RAG_SESSIONS.get(worker_id)
    if not session:
        return json.dumps({"status": "error", "message": "No active RAG session. Call setup_rag_engine first."})

    index_path = session.get("index_path")
    
    # [模拟检索逻辑]
    # 实际代码中，这里会调用 LangChain 或 LlamaIndex 的检索接口
    # 这里我们简单读取文件模拟检索
    results = []
    try:
        if os.path.exists(index_path):
            with open(index_path, 'r', encoding='utf-8') as f:
                # 简单实现：逐行查找包含查询词的内容
                lines = f.readlines()
                for line in lines:
                    if len(results) >= top_k: break
                    if query.lower() in line.lower():
                        results.append(line.strip())
                
                # 如果没找到，为了演示返回前几行
                if not results and lines:
                    results = [l.strip() for l in lines[:top_k]]
        else:
            return json.dumps({"status": "error", "message": f"Index file not found: {index_path}"})

    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})

    return json.dumps({
        "status": "success",
        "results": results
    })

@mcp.tool()
async def release_rag_engine(worker_id: str) -> str:
    """释放 RAG 资源"""
    session = RAG_SESSIONS.get(worker_id)
    if session:
        resource_id = session.get("resource_id")
        async with httpx.AsyncClient() as client:
            try:
                await client.post(
                    f"{RESOURCE_API_URL}/release", 
                    json={"resource_id": resource_id, "worker_id": worker_id}, 
                    timeout=10
                )
            except:
                pass
        RAG_SESSIONS.pop(worker_id, None)
    return "Released"

if __name__ == "__main__":
    # RAG Server 运行在 8081 端口，避免与 OSWorld Server (8080) 冲突
    mcp.settings.debug = True
    mcp.settings.host = "0.0.0.0"
    mcp.settings.port = 8081
    
    print(f"🚀 Starting RAG MCP Server on {mcp.settings.host}:{mcp.settings.port} (SSE Mode)...")
    mcp.run(transport='sse')