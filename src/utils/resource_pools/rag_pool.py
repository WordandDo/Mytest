# src/utils/resource_pools/rag_pool.py
import logging
import os
import sys
import time
import uuid
import multiprocessing
import uvicorn
import traceback
from typing import Dict, Any, Type, Optional
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel

# 确保可以导入 envs 模块
cwd = os.getcwd()
if os.path.join(cwd, "src") not in sys.path:
    sys.path.append(os.path.join(cwd, "src"))

from utils.resource_pools.base import AbstractPoolManager, ResourceEntry, ResourceStatus
# 直接导入底层索引实现，不再依赖 RAGEnvironment
from utils.rag_index import get_rag_index_class, BaseRAGIndex

logger = logging.getLogger(__name__)

# =========================================================================
# [Embedded RAG Server] 嵌入式 RAG 服务端逻辑
# =========================================================================
rag_server_app = FastAPI(title="Embedded RAG Service")
rag_index_instance: Optional[BaseRAGIndex] = None

# [新增] 全局配置对象，用于存储从 deployment_config 传来的默认值
SERVER_CONFIG = {
    "default_top_k": 5  # 默认兜底
}

class QueryRequest(BaseModel):
    query: str
    # [修改] 这里设为 Optional，如果为 None 则使用 SERVER_CONFIG 中的值
    top_k: Optional[int] = None 
    token: Optional[str] = None

@rag_server_app.post("/query")
async def api_query_index(request: QueryRequest):
    if not rag_index_instance:
        raise HTTPException(status_code=503, detail="Index not loaded yet")
    try:
        # [关键逻辑] 优先使用请求参数 -> 其次使用配置文件 -> 最后兜底 5
        effective_k = request.top_k if request.top_k is not None else SERVER_CONFIG["default_top_k"]
        
        results = rag_index_instance.query(request.query, top_k=effective_k)
        return {"status": "success", "results": results}
    except Exception as e:
        logger.error(f"RAG Query Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@rag_server_app.get("/health")
async def health_check():
    return {"status": "ok", "ready": rag_index_instance is not None}

def start_rag_server(port: int, config: Dict[str, Any]):
    """
    [子进程入口] 启动 RAG 服务
    config 参数就是 deployment_config.json 中的 "config" 部分
    """
    # 1. 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - [RAG-Server] - %(levelname)s - %(message)s')
    server_logger = logging.getLogger("RAG-Server")
    server_logger.info(f"Starting Embedded RAG Server on port {port}...")

    # 2. [关键] 将配置注入全局变量
    if "default_top_k" in config:
        SERVER_CONFIG["default_top_k"] = int(config["default_top_k"])
        server_logger.info(f"Configured default_top_k = {SERVER_CONFIG['default_top_k']}")

    # 3. 加载索引
    global rag_index_instance
    try:
        # 完整读取所有 deployment_config 参数
        kb_path = config.get("rag_kb_path", "")
        index_path = config.get("rag_index_path", "")
        model_name = config.get("rag_model_name", "sentence-transformers/all-MiniLM-L6-v2")
        use_faiss = config.get("use_faiss", False)
        device = config.get("embedding_device", "cpu")
        
        # [适配] 处理 use_gpu_index (兼容字符串 "true"/"false" 和布尔值)
        use_gpu_index_raw = config.get("use_gpu_index", False)
        if isinstance(use_gpu_index_raw, str):
            use_gpu_index = use_gpu_index_raw.lower() in ('true', '1', 'yes')
        else:
            use_gpu_index = bool(use_gpu_index_raw)

        # 实例化
        IndexClass = get_rag_index_class(use_faiss=use_faiss)
        
        # 准备加载参数 (专门处理 RAGIndexLocal_faiss 的特殊参数)
        load_kwargs = {
            "index_path": index_path,
            "model_name": model_name,
            "device": device
        }
        # 如果是 Faiss 索引，注入 gpu 配置
        if "faiss" in IndexClass.__name__.lower():
            load_kwargs["use_gpu_index"] = use_gpu_index
            server_logger.info(f"GPU Index Enabled: {use_gpu_index}")

        # 加载或构建逻辑
        if index_path and os.path.exists(os.path.join(index_path, "metadata.json")):
            server_logger.info(f"Loading existing index from {index_path}...")
            rag_index_instance = IndexClass.load_index(**load_kwargs)
        else:
            server_logger.info(f"Building new index from {kb_path}...")
            # 构建时的参数
            build_init_kwargs = {"model_name": model_name, "device": device}
            if "faiss" in IndexClass.__name__.lower():
                build_init_kwargs["use_gpu_index"] = use_gpu_index
            
            rag_index_instance = IndexClass(**build_init_kwargs)
            rag_index_instance.build_index(file_path=kb_path)
            
            if index_path:
                rag_index_instance.save_index(index_path)
        
        server_logger.info("✅ Index loaded successfully.")
        
    except Exception as e:
        server_logger.error(f"Failed to load index: {e}")
        traceback.print_exc()
    
    # 4. 启动 uvicorn
    uvicorn.run(rag_server_app, host="0.0.0.0", port=port, log_level="warning")

# =========================================================================
# [Pool Manager] 资源池管理逻辑
# =========================================================================
class RAGPoolImpl(AbstractPoolManager):
    """
    RAG 资源池实现 (Process Manager 模式)
    负责启动/停止 RAG 子进程，并分配连接信息。
    """
    def __init__(self, 
                 num_rag_workers: int = 2,
                 rag_service_port: int = 8001,  # [配置] 服务端口
                 **kwargs):
        super().__init__(num_items=num_rag_workers)
        self.service_port = int(os.environ.get("RAG_SERVICE_PORT", rag_service_port))
        self.service_url = f"http://localhost:{self.service_port}"
        self.server_process: Optional[multiprocessing.Process] = None
        self.rag_config = kwargs  # 保存配置传给子进程

    def initialize_pool(self, max_workers: int = 10) -> bool:
        """启动 RAG 子进程"""
        logger.info(f"Initializing RAG Pool (Starting Subprocess on port {self.service_port})...")
        
        # 1. 启动子进程
        self.server_process = multiprocessing.Process(
            target=start_rag_server,
            args=(self.service_port, self.rag_config),
            daemon=True
        )
        self.server_process.start()
        
        # 2. 等待服务就绪 (简单的轮询检查)
        import requests
        retries = 30 # 30秒超时
        logger.info("Waiting for RAG Server to be ready...")
        for _ in range(retries):
            try:
                resp = requests.get(f"{self.service_url}/health", timeout=1)
                if resp.status_code == 200 and resp.json().get("ready"):
                    logger.info("✅ RAG Server is ready and serving.")
                    break
            except Exception:
                pass
            time.sleep(1)
        else:
            logger.warning("⚠️ RAG Server started but health check failed (might still be loading index).")

        # 3. 初始化逻辑资源槽位
        return super().initialize_pool(max_workers)

    def _create_resource(self, index: int) -> ResourceEntry:
        """创建逻辑连接凭证"""
        return ResourceEntry(
            resource_id=f"rag-session-{index}",
            status=ResourceStatus.FREE,
            config={
                "token": str(uuid.uuid4()),
                "base_url": self.service_url  # 注入直连地址
            }
        )

    def _validate_resource(self, entry: ResourceEntry) -> bool:
        # 只要子进程活着，资源就有效
        return self.server_process is not None and self.server_process.is_alive()

    def _get_connection_info(self, entry: ResourceEntry) -> Dict[str, Any]:
        """返回直连信息给 MCP Server"""
        return {
            "id": entry.resource_id,
            "type": "rag_service",
            "base_url": entry.config["base_url"], # 直连 URL
            "token": entry.config["token"],
            "status": "ready"
        }

    def _reset_resource(self, entry: ResourceEntry) -> None:
        pass

    def _stop_resource(self, entry: ResourceEntry) -> None:
        pass

    def stop_all(self) -> None:
        """停止所有资源时，杀掉子进程"""
        super().stop_all()
        if self.server_process and self.server_process.is_alive():
            logger.info("Stopping RAG Server process...")
            self.server_process.terminate()
            self.server_process.join()