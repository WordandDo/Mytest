import os
from typing import Any, Dict, Optional, Type

from .enviroment import Environment
from .rag_index import (
    BaseRAGIndex,
    RAGIndexLocal,
    RAGIndexLocal_faiss,
    get_rag_index_class,
)

__all__ = [
    "RAGIndexLocal",
    "RAGIndexLocal_faiss",
    "get_rag_index_class",
    "RAGEnvironment",
]


class RAGEnvironment(Environment):
    """RAG environment that manages local embedding indexes."""

    def __init__(
        self,
        kb_path: Optional[str] = None,
        index_path: Optional[str] = None,
        emb_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        emb_batchsize: int = 64,
        use_faiss: bool = False,
        load_index: bool = True,
        embedding_device: str = "cuda",
        max_seq_length: int = 512,
        use_gpu_index: bool = False,
        max_chunks: Optional[int] = None,
        num_workers: int = 1,
        **kwargs: Any,
    ) -> None:
        self.kb_path = kb_path
        self.index_path = index_path
        self.emb_model = emb_model
        self.emb_batchsize = emb_batchsize
        self.use_faiss = use_faiss
        self.load_index_flag = load_index
        self.embedding_device = embedding_device
        self.max_seq_length = max_seq_length
        self.use_gpu_index = use_gpu_index
        self.max_chunks = max_chunks
        self.num_workers = num_workers

        self.rag_index: Optional[BaseRAGIndex] = None
        self._rag_tools_initialized = False

        super().__init__(**kwargs)

        self._setup_rag_index()
        self._initialize_tools()

    @property
    def mode(self) -> str:
        return "rag"

    def _setup_rag_index(self) -> None:
        """Load an existing index or build a new one."""
        IndexClass: Type[BaseRAGIndex] = get_rag_index_class(use_faiss=self.use_faiss)

        should_try_load = bool(self.index_path and (self.load_index_flag or not self.kb_path))
        last_load_error: Optional[Exception] = None

        if should_try_load:
            try:
                print(f"尝试从 {self.index_path} 加载现有索引...")
                load_kwargs = {
                    "index_path": self.index_path,
                    "model_name": self.emb_model,
                    "device": self.embedding_device,
                    "max_seq_length": self.max_seq_length,
                }
                if issubclass(IndexClass, RAGIndexLocal_faiss):
                    load_kwargs["use_gpu_index"] = self.use_gpu_index

                self.rag_index = IndexClass.load_index(**load_kwargs)
                print(f"✓ 成功加载索引,包含 {len(self.rag_index.chunks)} 个文档块")
                return
            except FileNotFoundError as exc:
                print("未找到现有索引,将构建新索引")
                last_load_error = exc
            except Exception as exc:  # pylint: disable=broad-except
                print(f"加载索引失败: {exc}, 将构建新索引")
                import traceback

                traceback.print_exc()
                last_load_error = exc

        if not self.kb_path:
            error_msg = "未提供 kb_path, 无法构建新的 RAG 索引"
            if last_load_error:
                error_msg += f"；同时无法加载现有索引 ({last_load_error})"
            raise ValueError(error_msg) from last_load_error

        if not os.path.exists(self.kb_path):
            raise FileNotFoundError(f"知识库文件不存在: {self.kb_path}")

        print(f"\n从 {self.kb_path} 构建新的RAG索引...")

        init_kwargs = {
            "model_name": self.emb_model,
            "device": self.embedding_device,
            "max_seq_length": self.max_seq_length,
        }
        if issubclass(IndexClass, RAGIndexLocal_faiss):
            init_kwargs["use_gpu_index"] = self.use_gpu_index

        self.rag_index = IndexClass(**init_kwargs)
        self.rag_index.build_index(
            file_path=self.kb_path,
            batch_size=self.emb_batchsize,
            max_chunks=self.max_chunks,
            num_workers=self.num_workers,
        )

        print(f"✓ 成功构建索引,包含 {len(self.rag_index.chunks)} 个文档块")

        if self.index_path:
            print(f"\n保存索引到 {self.index_path}...")
            self.rag_index.save_index(self.index_path)
            print("✓ 索引已保存")

    def _initialize_tools(self) -> None:
        """Register RAG-specific tools once the index is ready."""
        if self.rag_index is None:
            self._rag_tools_initialized = False
            return

        if self._rag_tools_initialized:
            return

        try:
            from tools.rag_tools import QueryRAGIndexTool

            local_search_tool = QueryRAGIndexTool(self.rag_index)
            self.register_tool(local_search_tool)
            self._rag_tools_initialized = True
        except ImportError as exc:
            raise ImportError("RAG工具不可用") from exc

    def get_index_info(self) -> Dict[str, Any]:
        """Return diagnostic information about the current index."""
        if self.rag_index is None:
            return {"status": "not_initialized"}

        return {
            "status": "initialized",
            "num_chunks": len(self.rag_index.chunks) if hasattr(self.rag_index, "chunks") else 0,
            "kb_path": self.kb_path,
            "index_path": self.index_path,
            "emb_model": self.emb_model,
            "use_faiss": self.use_faiss,
            "use_gpu_index": self.use_gpu_index,
            "embedding_device": self.embedding_device,
        }
