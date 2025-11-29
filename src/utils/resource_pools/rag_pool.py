# src/utils/resource_pools/rag_pool.py
import logging
import os
import sys
import traceback
from typing import Dict, Any, Type

# 确保可以导入 envs 模块
cwd = os.getcwd()
if os.path.join(cwd, "src") not in sys.path:
    sys.path.append(os.path.join(cwd, "src"))

from utils.resource_pools.base import AbstractPoolManager, ResourceEntry, ResourceStatus
# 直接导入底层索引实现，不再依赖 RAGEnvironment
from utils.rag_index import get_rag_index_class, BaseRAGIndex, RAGIndexLocal_faiss

logger = logging.getLogger(__name__)

class RAGPoolImpl(AbstractPoolManager):
    """
    RAG 资源池实现。
    
    职责:
    1. 生命周期管理: 在初始化时负责检查、构建或加载 RAG 索引。
    2. 资源分配: 管理 Worker 对索引的并发访问权限。
    """
    def __init__(self, 
                 num_rag_workers: int = 2, 
                 rag_kb_path: str = "src/data/rag_demo.jsonl",
                 rag_index_path: str = "src/data/rag_index_storage",
                 rag_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 use_faiss: bool = False,
                 embedding_device: str = "cpu",
                 **kwargs):
        super().__init__(num_items=num_rag_workers)
        
        self.kb_path = rag_kb_path
        self.index_path = rag_index_path
        self.model_name = rag_model_name
        self.device = embedding_device
        self.use_faiss = use_faiss
        self.kwargs = kwargs
        
        # 存储加载后的索引实例（可选，如果仅作路径分发可不存，但为了验证索引完整性建议加载一次）
        self.rag_index_instance = None
        
        # === 初始化生命周期管理 ===
        self._initialize_rag_index()

    def _initialize_rag_index(self):
        """核心逻辑：检查、加载或构建索引"""
        logger.info(f"Initializing RAG Pool (KB: {self.kb_path})...")
        
        # 获取索引类 (NumPy 或 Faiss)
        IndexClass: Type[BaseRAGIndex] = get_rag_index_class(use_faiss=self.use_faiss)
        
        # 1. 尝试加载现有索引
        if self.index_path and os.path.exists(os.path.join(self.index_path, "metadata.json")):
            try:
                logger.info(f"Attempting to load existing index from {self.index_path}...")
                load_kwargs = {
                    "index_path": self.index_path,
                    "model_name": self.model_name,
                    "device": self.device,
                    "max_seq_length": int(self.kwargs.get("max_seq_length", 512))
                }
                if issubclass(IndexClass, RAGIndexLocal_faiss):
                    use_gpu_index = self.kwargs.get("use_gpu_index", False)
                    # 确保 use_gpu_index 是布尔值
                    if isinstance(use_gpu_index, str):
                        use_gpu_index = use_gpu_index.lower() in ('true', '1', 'yes')
                    load_kwargs["use_gpu_index"] = bool(use_gpu_index)

                self.rag_index_instance = IndexClass.load_index(**load_kwargs)
                logger.info(f"✅ Successfully loaded index with {len(self.rag_index_instance.chunks)} chunks.")
                return
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}. Will attempt to rebuild.")
                traceback.print_exc()

        # 2. 如果加载失败或不存在，构建新索引
        if not self.kb_path or not os.path.exists(self.kb_path):
            raise FileNotFoundError(f"Knowledge base file not found: {self.kb_path}. Cannot build index.")

        logger.info(f"Building new RAG index from {self.kb_path}...")
        try:
            init_kwargs = {
                "model_name": self.model_name,
                "device": self.device,
                "max_seq_length": int(self.kwargs.get("max_seq_length", 512))
            }
            if issubclass(IndexClass, RAGIndexLocal_faiss):
                use_gpu_index = self.kwargs.get("use_gpu_index", False)
                # 确保 use_gpu_index 是布尔值
                if isinstance(use_gpu_index, str):
                    use_gpu_index = use_gpu_index.lower() in ('true', '1', 'yes')
                init_kwargs["use_gpu_index"] = bool(use_gpu_index)

            # 初始化并构建
            self.rag_index_instance = IndexClass(**init_kwargs)
            self.rag_index_instance.build_index(
                file_path=self.kb_path,
                batch_size=int(self.kwargs.get("emb_batchsize", 64)),
                max_chunks=self.kwargs.get("max_chunks", None),
                num_workers=int(self.kwargs.get("num_workers", 1))
            )
            
            # 3. 保存索引
            if self.index_path:
                logger.info(f"Saving index to {self.index_path}...")
                self.rag_index_instance.save_index(self.index_path)
                logger.info("✅ Index built and saved.")
                
        except Exception as e:
            logger.error(f"Critical error building RAG index: {e}")
            raise RuntimeError("RAG Pool initialization failed") from e

    def _create_resource(self, index: int) -> ResourceEntry:
        """创建一个 RAG Worker 槽位"""
        resource_id = f"rag-worker-{index}"
        
        # 配置信息传递给申请者
        config = {
            "index_path": self.index_path,
            "kb_path": self.kb_path,
            "emb_model": self.model_name,
            "device": self.device,
            "use_faiss": self.use_faiss
        }
        
        return ResourceEntry(
            resource_id=resource_id,
            status=ResourceStatus.FREE,
            config=config
        )

    def _validate_resource(self, entry: ResourceEntry) -> bool:
        """验证索引目录是否存在"""
        return os.path.exists(self.index_path)

    def _get_connection_info(self, entry: ResourceEntry) -> Dict[str, Any]:
        """返回给 Worker 的连接/配置信息"""
        return {
            "id": entry.resource_id,
            "type": "rag",
            "index_path": entry.config["index_path"],
            "kb_path": entry.config["kb_path"],
            "emb_model": entry.config["emb_model"],
            "use_faiss": entry.config["use_faiss"], # 传递是否使用 Faiss 的信息
            "status": "ready"
        }

    def _reset_resource(self, entry: ResourceEntry) -> None:
        pass

    def _stop_resource(self, entry: ResourceEntry) -> None:
        pass