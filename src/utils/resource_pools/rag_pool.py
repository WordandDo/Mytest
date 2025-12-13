# src/utils/resource_pools/rag_pool.py
import logging
import os
import sys
import time
import uuid
import multiprocessing
import threading
import uvicorn
import traceback
import signal
import subprocess
from queue import Queue
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

LOG_DIR = os.path.join(cwd, "logs")
RAG_PID_FILE = os.path.join(LOG_DIR, "rag_server.pid")

def _write_rag_pid(pid: int):
    """Persist the current RAG server PID for external tooling."""
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(RAG_PID_FILE, "w", encoding="utf-8") as fh:
            fh.write(str(pid))
    except Exception as e:
        logger.warning(f"Failed to write RAG PID file: {e}")

def _remove_rag_pid_file():
    """Helper to clear the persisted RAG pid when the service stops."""
    try:
        os.remove(RAG_PID_FILE)
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning(f"Failed to remove RAG PID file: {e}")

# =========================================================================
# [Embedded RAG Server] 嵌入式 RAG 服务端逻辑
# =========================================================================
rag_server_app = FastAPI(title="Embedded RAG Service")
rag_index_instance: Optional[BaseRAGIndex] = None

# [新增] 全局配置对象，用于存储从 deployment_config 传来的默认值
SERVER_CONFIG = {
    "default_top_k": 5  # 默认兜底
}

# --- 全局状态管理 ---
loading_state = {
    "status": "initializing",
    "ready": False,
    "error": None,
    "progress": "Starting..."
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
    search_type: str = "dense"  # 新增：检索类型，支持 "sparse" 或 "dense"

@rag_server_app.post("/query")
async def api_query_index(request: QueryRequest):
    if not rag_index_instance:
        raise HTTPException(status_code=503, detail="Index not loaded yet")
    try:
        # [关键逻辑] 优先使用请求参数 -> 其次使用配置文件 -> 最后兜底 5
        effective_k = request.top_k if request.top_k is not None else SERVER_CONFIG["default_top_k"]

        # [新增] 传递 search_type 参数给索引的 query 方法
        results = rag_index_instance.query(
            request.query,
            top_k=effective_k,
            search_type=request.search_type
        )
        return {"status": "success", "results": results}
    except Exception as e:
        logger.error(f"RAG Query Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@rag_server_app.get("/health")
async def health_check():
    """
    严格的健康检查：只有当后台完全加载完毕(ready=True)时才返回 ok
    """
    if loading_state["error"]:
        # 如果后台崩了，直接报错
        return {
            "status": "error",
            "ready": False,
            "detail": loading_state["error"]
        }

    if loading_state["ready"]:
        # 只有这里才返回 True
        return {
            "status": "ok",
            "ready": True,
            "detail": "Service is fully ready"
        }

    # 否则一直返回 False，让脚本继续转圈等待
    return {
        "status": "loading",
        "ready": False,
        "detail": loading_state["progress"]
    }

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

    # 4. 启动后台线程加载索引
    loader_thread = threading.Thread(
        target=_background_load_index, 
        args=(config.copy(),), 
        daemon=True
    )
    loader_thread.start()

    # 5. 立即启动 uvicorn
    uvicorn.run(rag_server_app, host="0.0.0.0", port=port, log_level="warning")


def _background_load_index(config: Dict[str, Any]):
    """
    [后台线程] 异步加载 RAG 索引，并在完成后设置全局实例。
    对 HybridRAGIndex 类型会执行 warmup() 预热。
    """
    global rag_index_instance
    try:
        logging.info("🧵 [Background] Starting index loading logic...")
        loading_state["progress"] = "Loading configuration..."

        # 提取基础路径配置
        kb_path = config.get("rag_kb_path", "")
        index_path = config.get("rag_index_path", "")
        model_name = config.get("rag_model_name", "sentence-transformers/all-MiniLM-L6-v2")
        device = config.get("embedding_device", "cpu")

        # 提取类型开关 (Boolean)
        def parse_bool(key, default=False):
            val = config.get(key, default)
            if isinstance(val, str):
                return val.lower() in ('true', '1', 'yes')
            return bool(val)

        use_faiss = parse_bool("use_faiss", False)
        use_gpu_index = parse_bool("use_gpu_index", False)
        # [新增] 紧凑型索引开关
        use_compact = parse_bool("use_compact", False)
        # [新增] 混合检索开关（替代 GainRAG）
        use_hybrid = parse_bool("use_hybrid", False)

        # 提取高级参数
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

        # [新增] Hybrid 混合检索特定参数
        bm25_index_path = config.get("bm25_index_path")  # BM25 索引路径
        dense_index_path = config.get("dense_index_path")  # Dense 索引路径（可选，默认用 index_path）
        corpus_path = config.get("corpus_path")  # 语料库路径（Dense 必需）

        # 调用新的工厂函数
        IndexClass = get_rag_index_class(
            use_faiss=use_faiss,
            use_compact=use_compact,
            use_hybrid=use_hybrid
        )
        logging.info(f"Selected Index Class: {IndexClass.__name__}")

        # 构建通用参数字典
        common_kwargs = {
            "model_name": model_name,
            "device": device,
            "embedding_devices": embedding_devices, # 传递多卡列表
        }

        # 针对 Faiss 体系的参数注入
        if "faiss" in IndexClass.__name__.lower():
            common_kwargs["use_gpu_index"] = use_gpu_index
            if gpu_parallel_degree:
                common_kwargs["gpu_parallel_degree"] = gpu_parallel_degree

        # 针对 Compact 索引的参数注入
        if "compact" in IndexClass.__name__.lower():
            if target_bytes:
                common_kwargs["target_bytes_per_vector"] = target_bytes
            # 开启内存映射以减少内存占用
            common_kwargs["memory_map"] = True

        # 针对 Hybrid 索引的参数注入
        if use_hybrid:
            if bm25_index_path:
                common_kwargs["bm25_index_path"] = bm25_index_path
            if dense_index_path:
                common_kwargs["dense_index_path"] = dense_index_path
            if corpus_path:
                common_kwargs["corpus_path"] = corpus_path

        # 检查是否存在 metadata.json (标准 RAG)
        has_metadata = index_path and os.path.exists(os.path.join(index_path, "metadata.json"))

        should_load = has_metadata or use_hybrid  # Hybrid 模式总是使用懒加载

        # === 核心修改点：加载索引并调用 warmup ===
        if "Hybrid" in IndexClass.__name__:
            loading_state["progress"] = "Loading Hybrid Index components..."
            logging.info("⚡ Detected HybridRAGIndex, starting instantiation...")

            # 1. 实例化 (此时是懒加载，还没真正读文件)
            rag_index_instance = IndexClass.load_index(index_path=index_path, **common_kwargs)

            # 2. 调用 warmup 方法预热整个索引
            # 在这行执行完之前，loading_state["ready"] 依然是 False
            loading_state["progress"] = "Warming up Hybrid Index (this may take several minutes)..."
            rag_index_instance.warmup()

        else:
            # 常规索引的加载逻辑
            if should_load:
                loading_state["progress"] = f"Loading existing index from {index_path}..."
                logging.info(f"Loading existing index from {index_path}...")
                rag_index_instance = IndexClass.load_index(
                    index_path=index_path,
                    **common_kwargs
                )
            else:
                if use_hybrid:
                    raise RuntimeError("HybridRAGIndex 需要预先构建的 BM25 和 Dense 索引")

                loading_state["progress"] = f"Building new index from {kb_path}..."
                logging.info(f"Building new index from {kb_path}...")
                rag_index_instance = IndexClass(**common_kwargs)
                rag_index_instance.build_index(
                    file_path=kb_path,
                    num_workers=0
                )

                if index_path:
                    rag_index_instance.save_index(index_path)

        # === 只有代码跑到这里，才宣布就绪 ===
        logging.info("✅ Index loading and warmup COMPLETED.")
        loading_state["ready"] = True
        loading_state["status"] = "ok"
        loading_state["progress"] = "Done"

    except Exception as e:
        error_msg = str(e)
        # 强制打印堆栈跟踪，确保即使日志丢失也能看到
        import traceback
        traceback.print_exc(file=sys.stderr)
        
        logging.critical(f"❌ [Background] Critical failure: {error_msg}", exc_info=True)
        loading_state["ready"] = False
        loading_state["status"] = "error"
        loading_state["error"] = error_msg
        # 如果加载失败，子进程应该退出
        sys.exit(1)

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
        self.is_recovering = False
        self.recovery_lock = threading.Lock()
        # 自动重启 RAG 服务会导致卡死，默认关闭
        self.enable_restart = False
        self._restart_notice_logged = False

    def initialize_pool(self, max_workers: int = 10) -> bool:
        """启动 RAG 子进程"""
        logger.info(f"Initializing RAG Pool (Starting Subprocess on port {self.service_port})...")
        _remove_rag_pid_file()
        
        # 1. 启动子进程
        self.server_process = multiprocessing.Process(
            target=start_rag_server,
            args=(self.service_port, self.rag_config),
            daemon=True
        )
        self.server_process.start()
        if self.server_process.pid:
            _write_rag_pid(self.server_process.pid)
        
        # 2. 等待服务就绪 (简单的轮询检查)
        import requests
        retries = int(self.rag_config.get("server_start_retries", 30))
        
        logger.info(f"Waiting for RAG Server to be ready (timeout={retries}s)...")
        for _ in range(retries):
            try:
                resp = requests.get(f"{self.service_url}/health", timeout=1)
                if resp.status_code == 200 and resp.json().get("ready"):
                    logger.info("✅ RAG Server is ready and serving.")
                    break # 成功！跳出循环，跳过 else，继续执行下方代码
            except Exception:
                pass
            
            # [可选优化]：如果发现子进程已经死了，直接提前终止等待
            if not self.server_process.is_alive():
                logger.error("❌ Detected RAG subprocess died unexpectedly during initialization.")
                break 
                
            time.sleep(1)
        else:
            # [关键修改] 循环耗尽仍未成功：进入此分支
            logger.error(f"❌ RAG Server failed to start after {retries}s. Aborting initialization.")
            
            # 1. 打印子进程状态辅助调试
            if self.server_process.is_alive():
                logger.error("   Subprocess is still alive but unresponsive (Hanged/Loading slow).")
            else:
                logger.error(f"   Subprocess died with exit code: {self.server_process.exitcode}")

            # 2. 清理残局
            self.stop_all()
            
            # 3. 明确返回失败，阻止 super().initialize_pool() 执行
            #    这样就不会创建那 50 个虚假的资源条目了
            return False 

        # [双重保险] 如果子进程中途死了（通过上面的 break 跳出），这里再拦一道
        if not self.server_process.is_alive():
             logger.error("❌ RAG Server process is dead. Initialization failed.")
             self.stop_all()
             return False

        # 快速路径：只重启后端，不重新创建逻辑资源
        if max_workers == 0:
            self._reset_queue_after_restart()
            return True

        # 3. 初始化逻辑资源槽位 (只有真正成功才会执行到这里)
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
        """
        验证资源是否有效：
        1. 子进程必须存活
        2. 索引必须加载完成（health check 返回 ready=True）
        """
        if self.is_recovering:
            return False

        if not (self.server_process and self.server_process.is_alive()):
            return False

        # 检查索引是否就绪
        try:
            import requests
            resp = requests.get(f"{self.service_url}/health", timeout=1)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("ready", False)  # 只有 ready=True 才算有效
        except Exception as e:
            logger.debug(f"Health check failed for {entry.resource_id}: {e}")
            return False

        return False

    def _reset_queue_after_restart(self) -> None:
        """重启后恢复资源队列和状态。"""
        with self.pool_lock:
            new_queue: Queue = Queue()
            free_count = 0
            for entry in self.pool.values():
                entry.status = ResourceStatus.FREE
                entry.allocated_to = None
                entry.allocated_at = None
                entry.error_message = None
                new_queue.put(entry.resource_id)
                free_count += 1

            self.free_queue = new_queue
            self.stats["free"] = free_count
            self.stats["occupied"] = 0
            self.stats["total"] = len(self.pool)

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
        if self.is_recovering:
            return

        # 如果资源仍然健康，则无需重启
        if self._validate_resource(entry):
            return

        if not self.enable_restart:
            if not self._restart_notice_logged:
                logger.warning("RAG restart logic is disabled to avoid system hangs. Please restart the service manually if needed.")
                self._restart_notice_logged = True
            return

        # 非阻塞获取锁，避免重复触发
        if not self.recovery_lock.acquire(blocking=False):
            return

        try:
            if self.is_recovering:
                return

            logger.warning(f"🚨 RAG Backend failure detected by {entry.resource_id}. Triggering ASYNC RESTART...")
            self.is_recovering = True

            restart_thread = threading.Thread(
                target=self._background_restart_task,
                daemon=True,
                name="RAG-Restart-Thread",
            )
            restart_thread.start()
        finally:
            self.recovery_lock.release()

    def _stop_resource(self, entry: ResourceEntry) -> None:
        pass

    def _background_restart_task(self):
        logger.info("🔧 [Background] RAG Restart sequence initiated (This will take a while)...")
        try:
            self.stop_all()
            success = self.initialize_pool(max_workers=0)

            if success:
                logger.info("✅ [Background] RAG Server restarted and READY.")
            else:
                logger.error("❌ [Background] RAG Server restart failed.")
        except Exception as e:
            logger.error(f"❌ [Background] Restart exception: {e}", exc_info=True)
        finally:
            self.is_recovering = False

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
        _remove_rag_pid_file()
        self.is_recovering = False
