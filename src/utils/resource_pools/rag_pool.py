# src/utils/resource_pools/rag_pool.py
import logging
import os
import sys
import time
import uuid
import multiprocessing
import uvicorn
import traceback
import signal
import subprocess
from typing import Dict, Any, Type, Optional
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel

# 确保可以导入 envs 模块
cwd = os.getcwd()
if os.path.join(cwd, "src") not in sys.path:
    sys.path.append(os.path.join(cwd, "src"))

from utils.resource_pools.base import AbstractPoolManager, ResourceEntry, ResourceStatus
# 直接导入底层索引实现，不再依赖 RAGEnvironment
# 使用新的 rag_index_new.py 模块
from utils.rag_index_new import get_rag_index_class, BaseRAGIndex

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

def kill_port_process(port: int):
    """
    强制杀死占用指定端口的进程
    """
    try:
        # 查找占用端口的进程
        result = subprocess.run(['lsof', '-i', f':{port}', '-t'], 
                              capture_output=True, text=True)
        if result.stdout:
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid:
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                        logger.info(f"Terminated process {pid} occupying port {port}")
                    except ProcessLookupError:
                        pass  # 进程已经退出
                    except PermissionError:
                        logger.warning(f"Permission denied terminating process {pid}")
        else:
            logger.info(f"No process found occupying port {port}")
    except Exception as e:
        logger.warning(f"Failed to kill process on port {port}: {e}")

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
    适配了新的 rag_index_new.py，支持 GainRAG、Compact 索引和多 GPU 配置
    """
    # 1. 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - [RAG-Server] - %(levelname)s - %(message)s')
    server_logger = logging.getLogger("RAG-Server")
    server_logger.info(f"Starting Embedded RAG Server on port {port}...")

    # 2. 清理占用目标端口的进程
    kill_port_process(port)

    # 3. 注入全局配置
    if "default_top_k" in config:
        SERVER_CONFIG["default_top_k"] = int(config["default_top_k"])
        server_logger.info(f"Configured default_top_k = {SERVER_CONFIG['default_top_k']}")

    # 4. 加载索引
    global rag_index_instance
    try:
        # ==========================================
        # [适配点 1] 提取基础路径配置
        # ==========================================
        kb_path = config.get("rag_kb_path", "")
        index_path = config.get("rag_index_path", "")
        model_name = config.get("rag_model_name", "sentence-transformers/all-MiniLM-L6-v2")
        device = config.get("embedding_device", "cpu")

        # ==========================================
        # [适配点 2] 提取类型开关 (Boolean)
        # ==========================================
        def parse_bool(key, default=False):
            val = config.get(key, default)
            if isinstance(val, str):
                return val.lower() in ('true', '1', 'yes')
            return bool(val)

        use_faiss = parse_bool("use_faiss", False)
        use_gpu_index = parse_bool("use_gpu_index", False)
        # [新增] 紧凑型索引开关
        use_compact = parse_bool("use_compact", False)
        # [新增] GainRAG 开关
        use_gainrag = parse_bool("use_gainrag", False)

        # ==========================================
        # [适配点 3] 提取高级参数
        # ==========================================
        # [新增] GPU 并行度
        gpu_parallel_degree = config.get("gpu_parallel_degree")
        if gpu_parallel_degree:
            gpu_parallel_degree = int(gpu_parallel_degree)

        # [新增] 多卡 Embedding 设备列表 (解析 "cuda:0,cuda:1" 字符串)
        embedding_devices = config.get("embedding_devices")
        if isinstance(embedding_devices, str) and embedding_devices.strip():
            embedding_devices = [d.strip() for d in embedding_devices.split(",")]
        elif not isinstance(embedding_devices, list):
            embedding_devices = None

        # [新增] Compact 索引特定参数
        target_bytes = config.get("target_bytes_per_vector")
        target_bytes = int(target_bytes) if target_bytes else None

        # [新增] GainRAG 特定参数
        passages_path = config.get("passages_path") # GainRAG 必须提供
        gpu_id = int(config.get("gpu_id", 0))

        # ==========================================
        # [适配点 4] 调用新的工厂函数
        # ==========================================
        # 工厂函数现在接收更多参数来决定返回哪个类
        IndexClass = get_rag_index_class(
            use_faiss=use_faiss,
            use_compact=use_compact,
            use_gainrag=use_gainrag
        )
        server_logger.info(f"Selected Index Class: {IndexClass.__name__}")

        # ==========================================
        # [适配点 5] 构建通用参数字典
        # ==========================================
        # 这些参数在 load_index 和 __init__ 中大多是通用的
        common_kwargs = {
            "model_name": model_name,
            "device": device,
            "embedding_devices": embedding_devices, # 传递多卡列表
        }

        # 针对 Faiss 体系的参数注入
        if "faiss" in IndexClass.__name__.lower() or use_gainrag:
            common_kwargs["use_gpu_index"] = use_gpu_index
            if gpu_parallel_degree:
                common_kwargs["gpu_parallel_degree"] = gpu_parallel_degree

        # 针对 Compact 索引的参数注入
        if "compact" in IndexClass.__name__.lower():
            if target_bytes:
                common_kwargs["target_bytes_per_vector"] = target_bytes
            # 开启内存映射以减少内存占用
            common_kwargs["memory_map"] = True

        # 针对 GainRAG 的参数注入
        if use_gainrag:
            common_kwargs["gpu_id"] = gpu_id
            if passages_path:
                common_kwargs["passages_path"] = passages_path

        # ==========================================
        # [适配点 6] 加载或构建逻辑
        # ==========================================
        # 检查是否存在 metadata.json (标准 RAG) 或 index.faiss (GainRAG)
        has_metadata = index_path and os.path.exists(os.path.join(index_path, "metadata.json"))
        has_gainrag_index = index_path and os.path.exists(os.path.join(index_path, "index.faiss"))

        should_load = has_metadata or (use_gainrag and has_gainrag_index)

        if should_load:
            server_logger.info(f"Loading existing index from {index_path}...")
            # 调用 load_index 类方法
            rag_index_instance = IndexClass.load_index(
                index_path=index_path,
                **common_kwargs
            )
        else:
            if use_gainrag:
                raise RuntimeError("GainRAGIndex does not support online building. Please provide a valid index_path.")

            server_logger.info(f"Building new index from {kb_path}...")
            # 实例化对象
            # 注意：GainRAGIndex 不会走这里
            rag_index_instance = IndexClass(**common_kwargs)

            # 触发构建
            # 可以在这里根据文件大小自动决定是否使用 build_index_streaming
            # 简单起见，这里演示标准构建，但传入 num_workers 以利用新代码的多进程能力
            rag_index_instance.build_index(
                file_path=kb_path,
                num_workers=0 # 0 表示自动利用 CPU 核心数
            )

            if index_path:
                rag_index_instance.save_index(index_path)

        server_logger.info("✅ Index loaded successfully.")

    except Exception as e:
        server_logger.error(f"Failed to load index: {e}")
        traceback.print_exc()
        # 如果加载失败，子进程应该退出，以便 Pool 能够检测到并处理（或者留在那里让用户通过 health check 发现）
        sys.exit(1)

    # 5. 启动 uvicorn
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
        # [修改] 尝试从配置中读取 server_start_retries，默认为 30
        # 这里的 self.rag_config 就是 deployment_config.json 中 "config" 下的内容
        retries = int(self.rag_config.get("server_start_retries", 30))
        
        logger.info(f"Waiting for RAG Server to be ready (timeout={retries}s)...")
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
            self.server_process.join(timeout=5)  # 等待最多5秒
            
            # 如果进程仍未退出，则强制杀死
            if self.server_process.is_alive():
                logger.warning("RAG Server process did not terminate gracefully, forcing kill...")
                self.server_process.kill()
                self.server_process.join()
        
        # 额外清理：确保端口被释放
        kill_port_process(self.service_port)
