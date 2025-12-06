import os
import json
import math
import numpy as np
import struct
import csv
import torch
from typing import Dict, List, Any, Optional, Sequence
from tqdm import trange, tqdm
import sys
import gc
import pickle
from multiprocessing import Pool, cpu_count
import time
from abc import ABC, abstractmethod

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pdb

# Check if Faiss is available
_FAISS_AVAILABLE = True
try:
    import faiss
except ImportError:
    _FAISS_AVAILABLE = False

# Check if sentence_transformers is available
_SENTENCE_TRANSFORMERS_AVAILABLE = True
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️ sentence_transformers 未安装,本地embedding功能不可用")

try:
    from transformers import AutoTokenizer, AutoConfig, BertModel  # type: ignore
except ImportError:
    AutoTokenizer = None  # type: ignore
    AutoConfig = None  # type: ignore
    BertModel = None  # type: ignore

_GAINRAG_TRANSFORMERS_AVAILABLE = all(
    dependency is not None for dependency in (AutoTokenizer, AutoConfig, BertModel)
)


# ============================================================================
# Faiss 辅助函数
# ============================================================================

def _suggest_ivf_params(num_vectors: int,
                        max_nlist: int = 4096,
                        max_nprobe: int = 16) -> tuple[int, int]:
    """
    根据向量数量给出合适的IVF参数

    Returns:
        (nlist, nprobe)
    """
    if num_vectors <= 0:
        return 1, 1
    nlist = max(1, min(max_nlist, int(np.sqrt(num_vectors))))
    nprobe = max(1, min(max_nprobe, nlist))
    return nlist, nprobe


# ============================================================================
# 多进程辅助函数（必须在模块级别定义才能被pickle）
# ============================================================================

def _encode_chunk_worker(args):
    """
    多进程worker函数：编码一批文本
    必须在模块级别定义以支持pickle序列化
    
    Args:
        args: (chunk_texts, model_name, max_seq_length, batch_size_inner, worker_id)
    
    Returns:
        编码后的向量数组
    """
    chunk_texts, model_name, max_seq_length, batch_size_inner, worker_id = args
    
    import os
    pid = os.getpid()
    
    # 在worker进程中加载模型
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import time
    
    # 记录开始时间
    start_time = time.time()
    
    # 加载模型（这是最耗时的部分）
    print(f"  Worker {worker_id} (PID {pid}): 开始加载模型...")
    model = SentenceTransformer(model_name, device='cpu')
    model.max_seq_length = max_seq_length
    load_time = time.time() - start_time
    print(f"  Worker {worker_id} (PID {pid}): 模型加载完成，耗时 {load_time:.1f}秒")
    
    # 处理文本
    print(f"  Worker {worker_id} (PID {pid}): 开始处理 {len(chunk_texts):,} 条文本...")
    chunk_embeddings = []
    process_start = time.time()
    
    for i in range(0, len(chunk_texts), batch_size_inner):
        batch = chunk_texts[i:i+batch_size_inner]
        batch_emb = model.encode(
            batch,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False
        )
        chunk_embeddings.append(batch_emb)
    
    process_time = time.time() - process_start
    total_time = time.time() - start_time
    print(f"  Worker {worker_id} (PID {pid}): 处理完成，处理耗时 {process_time:.1f}秒，总耗时 {total_time:.1f}秒")
    
    return np.vstack(chunk_embeddings).astype(np.float32)


class DiskBasedChunks:
    """
    Lazy loading list-like object for large JSONL files.
    Uses an offset file to jump to specific lines without reading the whole file.
    """
    def __init__(self, jsonl_path: str, offset_path: Optional[str] = None):
        self.jsonl_path = jsonl_path
        if offset_path is None:
            # If path ends with .jsonl, replace it with .offsets, otherwise append
            if jsonl_path.endswith('.jsonl'):
                offset_path = jsonl_path[:-6] + ".offsets"
            else:
                offset_path = jsonl_path + ".offsets"
        self.offset_path = offset_path
        
        if not os.path.exists(self.jsonl_path):
             raise FileNotFoundError(f"JSONL file not found: {self.jsonl_path}")
        
        if not os.path.exists(self.offset_path):
             raise FileNotFoundError(f"Offset file not found: {self.offset_path}. Please run convert_index.py first.")
             
        self._offsets = self._load_offsets()
        self._file = open(self.jsonl_path, 'rb') # Binary mode for precise seek
        
    def _load_offsets(self) -> List[int]:
        with open(self.offset_path, 'rb') as f:
            data = f.read()
        # Unpack all offsets (unsigned long long, 8 bytes)
        count = len(data) // 8
        return list(struct.unpack(f'<{count}Q', data))
        
    def __len__(self):
        return len(self._offsets)
        
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # Support slicing (e.g. for batch retrieval)
            # Note: This will be slow if slice is large, but functional
            start, stop, step = idx.indices(len(self))
            results = []
            for i in range(start, stop, step):
                results.append(self[i])
            return results

        if idx < 0:
            idx += len(self)
            
        if idx < 0 or idx >= len(self._offsets):
            raise IndexError("DiskBasedChunks index out of range")
            
        offset = self._offsets[idx]
        self._file.seek(offset)
        line = self._file.readline()
        return json.loads(line.decode('utf-8'))
        
    def __del__(self):
        if hasattr(self, '_file'):
            self._file.close()

# ============================================================================
# RAG 索引基类
# ============================================================================



class BaseRAGIndex(ABC):
    """
    RAG 索引公共基类，负责模型加载、文本解析与embedding生成。
    子类需实现向量存储、索引持久化及查询等特定逻辑。
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cuda", max_seq_length: int = 512,
                 embedding_devices: Optional[Sequence[str]] = None) -> None:
        if not _SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("需要安装 sentence_transformers: pip install sentence-transformers")

        self.model_name = model_name
        self.requested_device = device
        self.embedding_devices: List[str] = self._normalize_embedding_devices(embedding_devices)
        self._multi_gpu_devices: List[str] = list(self.embedding_devices) if len(self.embedding_devices) > 1 else []
        self._multi_gpu_failed = False

        primary_device = self.embedding_devices[0] if self.embedding_devices else device
        if device.lower().startswith("cpu") and self.embedding_devices:
            print("⚠️ 指定了 embedding_devices，但 device=CPU，将优先使用 GPU 设备进行编码")

        print(f"正在加载模型: {model_name}")
        self.model = SentenceTransformer(model_name, device=primary_device)
        model_max_length = self.model.get_max_seq_length()
        actual_max_length = min(max_seq_length, model_max_length)
        self.model.max_seq_length = actual_max_length
        if actual_max_length < max_seq_length:
            print(f"⚠️  警告: 模型最大支持长度为 {model_max_length}，已将 max_seq_length 从 {max_seq_length} 调整为 {actual_max_length}")
        print(f"✓ 模型已加载到 {primary_device}, 最大序列长度: {actual_max_length}")
        if self._multi_gpu_devices:
            print(f"  - 多GPU编码设备: {', '.join(self._multi_gpu_devices)}")

        self.chunks: List[Dict[str, Any]] = []
        self.device = primary_device
        self.max_seq_length = actual_max_length

    @staticmethod
    def _normalize_embedding_devices(devices: Optional[Sequence[str]]) -> List[str]:
        normalized: List[str] = []
        if not devices:
            return normalized
        seen: set[str] = set()
        for dev in devices:
            if dev is None:
                continue
            original = str(dev).strip()
            if not original:
                continue
            lowered = original.lower()
            if lowered == "cpu":
                continue
            if lowered == "cuda":
                token = "cuda:0"
            elif lowered.startswith("cuda:"):
                token = f"cuda:{lowered.split(':', 1)[1]}"
            elif lowered.startswith("cuda"):
                suffix = lowered[len("cuda") :].lstrip(":")
                token = f"cuda:{suffix}" if suffix else "cuda:0"
            elif lowered.isdigit():
                token = f"cuda:{lowered}"
            else:
                token = original
            if token not in seen:
                normalized.append(token)
                seen.add(token)
        return normalized

    @staticmethod
    def _normalize_embeddings(vectors: np.ndarray) -> np.ndarray:
        if vectors is None:
            raise ValueError("向量为空，无法归一化")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        return vectors / norms

    def _resolve_num_workers(self, num_workers: int) -> int:
        if num_workers == 0:
            if self.device == 'cpu':
                resolved = max(1, cpu_count() // 2)
                print(f"💡 自动设置worker数为CPU核心数的一半: {resolved}")
                return resolved
            return 1
        return num_workers

    def _combine_sample_text(self, sample: Dict[str, Any]) -> Optional[str]:
        if not isinstance(sample, dict):
            return None
        text = sample.get("text")
        if not isinstance(text, str) or not text:
            return None
        title = sample.get("title", "")
        return f"{title}\n\n{text}" if title else text

    def _prepare_embedding_texts(self) -> List[str]:
        print("正在合并title和text...")
        texts: List[str] = []
        for chunk in tqdm(self.chunks, desc="处理文本"):
            combined_text = self._combine_sample_text(chunk)
            if combined_text:
                texts.append(combined_text)
        print(f"✓ 文本合并完成")
        print()
        return texts

    def _load_kb_file(self, file_path: str) -> List[Any]:
        kb_content: List[Any] = []
        if file_path.endswith('.jsonl'):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="读取JSONL文件"):
                    if line.strip():
                        kb_content.append(json.loads(line))
        elif file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                kb_content = json.load(f)
        else:
            raise ValueError(f"不支持的文件格式: {file_path}")
        return kb_content

    def _extract_valid_chunks(self, kb_content: List[Any],
                              max_chunks: Optional[int]) -> List[Dict[str, Any]]:
        chunks = [
            sample for sample in kb_content
            if isinstance(sample, dict) and "text" in sample and
            isinstance(sample["text"], str) and sample["text"]
        ]
        if max_chunks:
            print(f"⚠️  限制最大文本块数: {max_chunks}")
            chunks = chunks[:max_chunks]
        if not chunks:
            raise ValueError("无法从文件中提取文本块")
        return chunks

    def _format_chunk_for_output(self, chunk: Any) -> str:
        if isinstance(chunk, dict):
            title = chunk.get("title", "")
            text = chunk.get("text", "")
            has_title = isinstance(title, str) and title.strip()
            has_text = isinstance(text, str) and text.strip()
            if has_title and has_text:
                return f"[{title}]\n{text}"
            if has_text:
                return text
            if has_title:
                return title
            return json.dumps(chunk, ensure_ascii=False)
        return str(chunk)

    def _get_embeddings_batch(self, texts: List[str], batch_size: int = 32,
                              show_progress: bool = True, num_workers: int = 1) -> np.ndarray:
        total_texts = len(texts)
        start_time = time.time()

        print(f"📊 Embedding生成配置:")
        print(f"  - 总文本数: {total_texts:,}")
        print(f"  - 批大小: {batch_size}")
        print(f"  - 设备: {self.device}")

        if self._multi_gpu_devices:
            print(f"  - 多GPU设备: {', '.join(self._multi_gpu_devices)}")
            if num_workers > 1:
                print("  ⚠️ 多GPU模式下忽略 num_workers 配置")
            embeddings_multi = None
            pool = None
            try:
                if not hasattr(self.model, "start_multi_process_pool") or not hasattr(self.model, "encode_multi_process"):
                    raise AttributeError("当前 sentence_transformers 版本不支持多进程编码")
                pool = self.model.start_multi_process_pool(target_devices=self._multi_gpu_devices)
                encode_kwargs = {
                    "batch_size": batch_size,
                    "normalize_embeddings": False,
                    "convert_to_numpy": True,
                }
                try:
                    embeddings_multi = self.model.encode_multi_process(
                        texts,
                        pool,
                        show_progress_bar=show_progress,
                        **encode_kwargs,
                    )
                except TypeError:
                    embeddings_multi = self.model.encode_multi_process(
                        texts,
                        pool,
                        **encode_kwargs,
                    )
            except Exception as exc:  # pylint: disable=broad-except
                print(f"⚠️ 多GPU编码失败({exc})，将回退到单GPU模式")
                self._multi_gpu_devices = []
                self._multi_gpu_failed = True
                embeddings_multi = None
            finally:
                if pool is not None:
                    try:
                        self.model.stop_multi_process_pool(pool)
                    except Exception:  # pylint: disable=broad-except
                        pass

            if embeddings_multi is not None:
                embeddings_np = np.asarray(embeddings_multi, dtype=np.float32)
                elapsed = time.time() - start_time
                speed = total_texts / elapsed if elapsed > 0 else 0.0
                print(f"\n✓ Embedding生成完成!")
                print(f"  - 耗时: {elapsed:.2f}秒")
                print(f"  - 速度: {speed:.1f} 文本/秒")
                print(f"  - 向量形状: {embeddings_np.shape}")
                return embeddings_np

        if self.device != 'cpu' and num_workers > 1:
            print("  ⚠️  GPU模式下不支持多进程，自动切换到单进程")
            num_workers = 1
        print(f"  - Worker数: {num_workers}")

        embeddings: List[np.ndarray] = []

        if num_workers > 1 and self.device == 'cpu':
            print(f"\n🚀 使用多进程加速 (workers={num_workers})...")
            print(f"💡 提示: 每个worker需要先加载模型（约10-30秒），请耐心等待...\n")

            chunk_size_per_worker = (total_texts + num_workers - 1) // num_workers
            text_chunks = [texts[i:i + chunk_size_per_worker] for i in range(0, total_texts, chunk_size_per_worker)]

            args_list = [
                (chunk, self.model_name, self.max_seq_length, batch_size, idx)
                for idx, chunk in enumerate(text_chunks)
            ]

            print(f"每个worker处理约 {chunk_size_per_worker:,} 条文本")
            print(f"开始启动 {num_workers} 个worker进程...\n")

            with Pool(processes=num_workers) as pool:
                results = []
                with tqdm(total=num_workers, desc="多进程处理", unit="worker",
                          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
                    for result in pool.imap(_encode_chunk_worker, args_list):
                        results.append(result)
                        pbar.update(1)

            embeddings = np.vstack(results).astype(np.float32)
            print(f"\n✓ 所有worker完成！")

        else:
            iterator = range(0, total_texts, batch_size)
            if show_progress:
                iterator = tqdm(
                    iterator,
                    desc="生成embeddings",
                    unit="batch",
                    total=(total_texts + batch_size - 1) // batch_size,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
                )

            for i in iterator:
                batch = texts[i:i + batch_size]
                batch_embeddings = self.model.encode(
                    batch,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=False
                )
                embeddings.append(batch_embeddings)

                if (i // batch_size) % 10 == 0:
                    gc.collect()

            embeddings = np.vstack(embeddings).astype(np.float32)

        elapsed = time.time() - start_time
        speed = total_texts / elapsed if elapsed > 0 else 0.0
        print(f"\n✓ Embedding生成完成!")
        print(f"  - 耗时: {elapsed:.2f}秒")
        print(f"  - 速度: {speed:.1f} 文本/秒")
        print(f"  - 向量形状: {embeddings.shape}")

        return embeddings

    def build_index(self, file_path: str, batch_size: int = 64,
                    max_chunks: Optional[int] = None, num_workers: int = 1) -> None:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        num_workers = self._resolve_num_workers(num_workers)

        print(f"\n{'='*60}")
        print(f"📂 步骤 1/4: 加载知识库")
        print(f"{'='*60}")
        print(f"文件路径: {file_path}")
        start_time = time.time()

        kb_content = self._load_kb_file(file_path)

        load_time = time.time() - start_time
        print(f"✓ 文件加载完成，耗时: {load_time:.2f}秒")

        print(f"\n{'='*60}")
        print(f"🔍 步骤 2/4: 提取文本块")
        print(f"{'='*60}")
        self.chunks = self._extract_valid_chunks(kb_content, max_chunks)
        print(f"✓ 成功提取 {len(self.chunks):,} 个文本块")

        print(f"\n{'='*60}")
        print(f"🤖 步骤 3/4: 生成Embeddings")
        print(f"{'='*60}")
        texts = self._prepare_embedding_texts()

        vectors = self._get_embeddings_batch(
            texts,
            batch_size=batch_size,
            show_progress=True,
            num_workers=num_workers
        )

        self._store_embeddings(vectors)

        print(f"\n{'='*60}")
        print(f"📐 步骤 4/4: 向量后处理")
        print(f"{'='*60}")
        self._finalize_embeddings()

        gc.collect()

    @abstractmethod
    def _store_embeddings(self, vectors: np.ndarray) -> None:
        """子类负责缓存或写入embedding向量。"""

    @abstractmethod
    def _finalize_embeddings(self) -> None:
        """子类负责向量的最终处理流程，例如归一化或构建索引。"""

    def _save_checkpoint(self, checkpoint_file: str, processed_lines: int,
                         processed_chunks: int) -> None:
        checkpoint = {
            "processed_lines": processed_lines,
            "processed_chunks": processed_chunks,
            "timestamp": time.time()
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)

    def build_index_streaming(self, *args, **kwargs) -> None:
        raise NotImplementedError("该索引类型未实现流式构建，请在子类中实现。")

    @abstractmethod
    def save_index(self, index_path: str) -> None:
        """子类负责索引持久化。"""

    @classmethod
    @abstractmethod
    def load_index(cls, index_path: str, model_name: Optional[str] = None,
                   device: str = "cuda", max_seq_length: int = 512, **kwargs) -> "BaseRAGIndex":
        """子类负责从磁盘加载索引。"""

    @abstractmethod
    def query(self, query: str, top_k: int = 3) -> str:
        """子类负责实现查询逻辑。"""


# ============================================================================
# 本地 RAG 索引实现
# ============================================================================

class RAGIndexLocal(BaseRAGIndex):
    """
    本地RAG索引实现,使用sentence_transformers进行embedding
    优化内存消耗和效率,适合大规模知识库
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 device: str = "cuda", max_seq_length: int = 512,
                 embedding_devices: Optional[Sequence[str]] = None):
        super().__init__(
            model_name=model_name,
            device=device,
            max_seq_length=max_seq_length,
            embedding_devices=embedding_devices
        )
        self.vectors = None
        self.normalized_vectors = None

    def _store_embeddings(self, vectors: np.ndarray) -> None:
        self.vectors = vectors.astype(np.float32)

    def _finalize_embeddings(self) -> None:
        self._normalize_vectors()

    def _normalize_vectors(self):
        """归一化向量以加速余弦相似度计算"""
        if self.vectors is not None:
            self.normalized_vectors = self._normalize_embeddings(self.vectors)
            print("✓ 向量已归一化")

    def build_index_streaming(self, file_path: str, index_path: str, 
                             batch_size: int = 64, chunk_size: int = 10000,
                             max_chunks: Optional[int] = None, num_workers: int = 1,
                             resume: bool = True):
        """
        流式构建索引，支持超大规模知识库（边加载边处理边保存）
        
        Args:
            file_path: 知识库文件路径 (仅支持jsonl格式)
            index_path: 索引保存目录
            batch_size: embedding生成的批量大小
            chunk_size: 每次处理的文本块数量（控制内存使用）
            max_chunks: 最大处理的文本块数量(用于测试),None表示处理全部
            num_workers: 并行处理的worker数量
            resume: 是否从上次中断处继续（支持断点续传）
        """
        if not file_path.endswith('.jsonl'):
            raise ValueError("流式构建仅支持JSONL格式文件")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 自动确定worker数量
        num_workers = self._resolve_num_workers(num_workers)
        
        os.makedirs(index_path, exist_ok=True)
        
        # 检查checkpoint
        checkpoint_file = os.path.join(index_path, "checkpoint.json")
        start_line = 0
        processed_chunks = 0
        
        if resume and os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                start_line = checkpoint.get("processed_lines", 0)
                processed_chunks = checkpoint.get("processed_chunks", 0)
            print(f"💾 从checkpoint恢复: 已处理 {processed_chunks:,} 个文本块")
        
        print(f"\n{'='*60}")
        print(f"🌊 流式索引构建模式")
        print(f"{'='*60}")
        print(f"文件路径: {file_path}")
        print(f"索引路径: {index_path}")
        print(f"每批处理: {chunk_size:,} 个文本块")
        print(f"Batch大小: {batch_size}")
        print(f"Worker数: {num_workers}")
        print(f"{'='*60}\n")
        
        total_start_time = time.time()
        batch_chunks = []
        batch_texts = []
        current_line = 0
        batch_num = processed_chunks // chunk_size
        
        # 打开文件并逐行读取
        with open(file_path, 'r', encoding='utf-8') as f:
            # 跳过已处理的行
            if start_line > 0:
                print(f"⏭️  跳过已处理的 {start_line:,} 行...")
                for _ in range(start_line):
                    next(f)
                current_line = start_line
            
            # 使用tqdm显示文件读取进度
            pbar = tqdm(desc="处理文本块", unit="块", initial=processed_chunks)
            
            for line in f:
                current_line += 1
                
                if not line.strip():
                    continue
                
                try:
                    sample = json.loads(line)
                    
                    # 验证数据格式
                    if not (isinstance(sample, dict) and "text" in sample and 
                           isinstance(sample["text"], str) and sample["text"]):
                        continue
                    
                    # 合并title和text
                    title = sample.get("title", "")
                    text = sample.get("text", "")
                    combined_text = f"{title}\n\n{text}" if title else text
                    
                    batch_chunks.append(sample)
                    batch_texts.append(combined_text)
                    
                    # 达到chunk_size或达到max_chunks限制时处理这一批
                    if len(batch_chunks) >= chunk_size or \
                       (max_chunks and processed_chunks + len(batch_chunks) >= max_chunks):
                        
                        # 处理当前批次
                        self._process_and_save_batch(
                            batch_chunks, batch_texts, batch_num,
                            index_path, batch_size, num_workers
                        )
                        
                        processed_chunks += len(batch_chunks)
                        pbar.update(len(batch_chunks))
                        batch_num += 1
                        
                        # 保存checkpoint
                        self._save_checkpoint(checkpoint_file, current_line, processed_chunks)
                        
                        # 清空批次
                        batch_chunks = []
                        batch_texts = []
                        gc.collect()
                        
                        # 检查是否达到max_chunks
                        if max_chunks and processed_chunks >= max_chunks:
                            print(f"\n⚠️  已达到最大文本块数限制: {max_chunks:,}")
                            break
                
                except json.JSONDecodeError as e:
                    print(f"⚠️  跳过无效JSON (行{current_line}): {e}")
                    continue
            
            # 处理剩余的批次
            if batch_chunks:
                self._process_and_save_batch(
                    batch_chunks, batch_texts, batch_num,
                    index_path, batch_size, num_workers
                )
                processed_chunks += len(batch_chunks)
                pbar.update(len(batch_chunks))
                self._save_checkpoint(checkpoint_file, current_line, processed_chunks)
            
            pbar.close()
        
        # 合并所有批次的索引
        print(f"\n{'='*60}")
        print(f"🔗 合并索引分片")
        print(f"{'='*60}")
        self._merge_index_shards(index_path, batch_num + 1)
        
        # 删除checkpoint（构建完成）
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        
        total_time = time.time() - total_start_time
        print(f"\n{'='*60}")
        print(f"✅ 流式索引构建完成!")
        print(f"{'='*60}")
        print(f"总文本块数: {processed_chunks:,}")
        print(f"总耗时: {total_time:.2f}秒")
        print(f"平均速度: {processed_chunks/total_time:.1f} 块/秒")
        print(f"索引路径: {index_path}")
        print(f"{'='*60}\n")
    
    def _process_and_save_batch(self, chunks: List[Dict], texts: List[str],
                                batch_num: int, index_path: str,
                                batch_size: int, num_workers: int):
        """处理并保存一个批次的数据"""
        print(f"\n📦 处理批次 #{batch_num} ({len(chunks):,} 个文本块)...")
        
        start_time = time.time()
        
        # 生成embeddings
        vectors = self._get_embeddings_batch(
            texts,
            batch_size=batch_size,
            show_progress=False,  # 外层已有进度条
            num_workers=num_workers
        )
        
        # 保存这个批次的数据
        batch_dir = os.path.join(index_path, f"batch_{batch_num:04d}")
        os.makedirs(batch_dir, exist_ok=True)
        
        # 保存chunks
        chunks_file = os.path.join(batch_dir, "chunks.json")
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False)
        
        # 保存vectors
        vectors_file = os.path.join(batch_dir, "vectors.npy")
        np.save(vectors_file, vectors)
        
        elapsed = time.time() - start_time
        speed = len(chunks) / elapsed
        print(f"  ✓ 批次 #{batch_num} 完成 - 耗时: {elapsed:.2f}秒, 速度: {speed:.1f} 块/秒")
    
    def _merge_index_shards(self, index_path: str, num_batches: int):
        """合并所有批次的索引分片"""
        print(f"正在合并 {num_batches} 个批次...")
        
        all_chunks = []
        all_vectors = []
        
        for batch_num in tqdm(range(num_batches), desc="加载批次"):
            batch_dir = os.path.join(index_path, f"batch_{batch_num:04d}")
            
            if not os.path.exists(batch_dir):
                continue
            
            # 加载chunks
            chunks_file = os.path.join(batch_dir, "chunks.json")
            if os.path.exists(chunks_file):
                with open(chunks_file, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                    all_chunks.extend(chunks)
            
            # 加载vectors
            vectors_file = os.path.join(batch_dir, "vectors.npy")
            if os.path.exists(vectors_file):
                vectors = np.load(vectors_file)
                all_vectors.append(vectors)
        
        # 合并vectors
        if all_vectors:
            self.vectors = np.vstack(all_vectors).astype(np.float32)
            self.chunks = all_chunks
            
            print(f"✓ 合并完成: {len(self.chunks):,} 个文本块")
            
            # 归一化向量
            print("正在归一化向量...")
            self._normalize_vectors()
            
            # 保存最终索引
            print("正在保存最终索引...")
            self.save_index(index_path)
            
            # 清理临时批次文件
            print("正在清理临时文件...")
            for batch_num in range(num_batches):
                batch_dir = os.path.join(index_path, f"batch_{batch_num:04d}")
                if os.path.exists(batch_dir):
                    import shutil
                    shutil.rmtree(batch_dir)
            
            print("✓ 临时文件清理完成")

    def save_index(self, index_path: str):
        """
        保存索引到磁盘
        
        Args:
            index_path: 保存索引的目录路径
        """
        if self.vectors is None or not self.chunks:
            raise ValueError("索引为空,无法保存。请先调用 build_index()")

        os.makedirs(index_path, exist_ok=True)
        
        # 保存文本块
        chunks_file = os.path.join(index_path, "chunks.json")
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        
        # 保存向量
        vectors_file = os.path.join(index_path, "vectors.npy")
        np.save(vectors_file, self.vectors)
        
        # 保存元数据
        metadata = {
            "model_name": self.model_name,
            "device": self.device,
            "max_seq_length": self.max_seq_length,
            "num_chunks": len(self.chunks),
            "vector_dim": self.vectors.shape[1],
            "embedding_devices": self.embedding_devices,
        }
        metadata_file = os.path.join(index_path, "metadata.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 索引已保存到: {index_path}")
        print(f"  - 文本块: {len(self.chunks)}")
        print(f"  - 向量维度: {self.vectors.shape}")

    @classmethod
    def load_index(cls, index_path: str, model_name: Optional[str] = None,
                   device: str = "cuda", max_seq_length: int = 512,
                   embedding_devices: Optional[Sequence[str]] = None, **kwargs):
        """
        从磁盘加载索引
        
        Args:
            index_path: 索引目录路径
            model_name: 模型名称(如果为None则从metadata读取)
            device: 运行设备
            max_seq_length: 最大序列长度
        
        Returns:
            加载的RAGIndexLocal实例
        """
        chunks_file = os.path.join(index_path, "chunks.json")
        vectors_file = os.path.join(index_path, "vectors.npy")
        metadata_file = os.path.join(index_path, "metadata.json")

        # 检查文件存在性
        if not os.path.exists(chunks_file):
            raise FileNotFoundError(f"未找到chunks文件: {chunks_file}")
        if not os.path.exists(vectors_file):
            raise FileNotFoundError(f"未找到vectors文件: {vectors_file}")
        
        # 读取元数据
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            if model_name is None:
                model_name = metadata.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
            if embedding_devices is None:
                embedding_devices = metadata.get("embedding_devices")
            print(f"✓ 从元数据读取配置: {metadata}")
        else:
            if model_name is None:
                model_name = "sentence-transformers/all-MiniLM-L6-v2"
            print("⚠️ 未找到元数据文件,使用默认配置")

        # 创建实例
        instance = cls(
            model_name=model_name,
            device=device,
            max_seq_length=max_seq_length,
            embedding_devices=embedding_devices,
        )
        
        # 加载数据
        with open(chunks_file, 'r', encoding='utf-8') as f:
            instance.chunks = json.load(f)
        
        instance.vectors = np.load(vectors_file)
        instance._normalize_vectors()
        
        print(f"✓ 成功加载索引: {len(instance.chunks)} 个文本块, 向量形状: {instance.vectors.shape}")
        
        return instance

    def query(self, query: str, top_k: int = 3) -> str:
        """
        查询索引
        
        Args:
            query: 查询文本
            top_k: 返回前k个结果
        
        Returns:
            检索到的上下文文本
        """
        if self.vectors is None:
            raise RuntimeError("索引未构建或加载,无法查询")
        
        if not query:
            raise ValueError("查询文本不能为空")

        # 生成查询向量
        query_vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False
        )[0].astype(np.float32)
        
        # 归一化查询向量
        query_norm = query_vector / np.linalg.norm(query_vector)
        
        # 计算相似度
        similarities = np.dot(self.normalized_vectors, query_norm)
        
        # 获取top_k结果
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]

        retrieved_chunks: List[Dict[str, Any]] = []
        for idx in top_k_indices:
            int_idx = int(idx)
            if 0 <= int_idx < len(self.chunks):
                retrieved_chunks.append(self.chunks[int_idx])
            else:
                print(f"⚠️ 查询结果索引越界: {int_idx} (chunks={len(self.chunks)})，已忽略")

        if not retrieved_chunks:
            return "[查询错误] 未检索到有效的文本块"

        formatted_chunks = [self._format_chunk_for_output(chunk) for chunk in retrieved_chunks]
        context = "\n---\n".join(formatted_chunks)

        return f"### Retrieved Context:\n{context}"



class RAGIndexLocal_faiss(BaseRAGIndex):
    """
    使用Faiss加速的本地RAG索引
    适合超大规模知识库(百万级以上)
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cuda", max_seq_length: int = 512,
                 use_gpu_index: bool = False,
                 gpu_parallel_degree: Optional[int] = None,
                 embedding_devices: Optional[Sequence[str]] = None):
        super().__init__(
            model_name=model_name,
            device=device,
            max_seq_length=max_seq_length,
            embedding_devices=embedding_devices,
        )
        
        if not _FAISS_AVAILABLE:
            raise ImportError("需要安装 faiss: pip install faiss-cpu 或 faiss-gpu")
        
        self.faiss_index = None
        self.use_gpu_index = use_gpu_index
        self.index_nlist: Optional[int] = None
        self.index_nprobe: Optional[int] = None
        self._faiss_on_gpu = False
        self._gpu_resources: Optional[List["faiss.StandardGpuResources"]] = None
        self._gpu_device_ids: Optional[List[int]] = None
        self._effective_gpu_parallel_degree: int = 0
        self.gpu_parallel_degree = 1
        if gpu_parallel_degree is not None:
            try:
                self.gpu_parallel_degree = max(1, int(gpu_parallel_degree))
            except (TypeError, ValueError):
                print(f"⚠️ 无效的gpu_parallel_degree({gpu_parallel_degree}), 已回退为1")
        if self.use_gpu_index and self.gpu_parallel_degree > 1:
            print(f"✓ Faiss索引GPU并行度: {self.gpu_parallel_degree}")
        self.vectors: Optional[np.ndarray] = None
        print(f"✓ Faiss索引模式: {'GPU' if use_gpu_index else 'CPU'}")

    def _store_embeddings(self, vectors: np.ndarray) -> None:
        prepared = np.ascontiguousarray(vectors.astype(np.float32))
        faiss.normalize_L2(prepared)
        self.vectors = prepared

    def _finalize_embeddings(self) -> None:
        print("✓ 向量已归一化并缓存，下一步将构建Faiss索引")

    def _create_cpu_ivf_index(self, vector_dim: int, total_vectors: int) -> "faiss.Index":
        total_vectors = max(1, total_vectors)
        nlist, nprobe = _suggest_ivf_params(total_vectors)
        print(f"正在创建Faiss IndexIVFFlat (维度={vector_dim}, nlist={nlist}, nprobe={nprobe})")
        quantizer = faiss.IndexFlatIP(vector_dim)
        index = faiss.IndexIVFFlat(quantizer, vector_dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.nprobe = nprobe
        self.index_nlist = nlist
        self.index_nprobe = nprobe
        return index

    def _prepare_gpu_resources(self) -> tuple[List["faiss.StandardGpuResources"], List[int]]:
        if not hasattr(faiss, "get_num_gpus"):
            raise RuntimeError("当前Faiss库未编译GPU支持")
        available_gpus = faiss.get_num_gpus()
        if available_gpus <= 0:
            raise RuntimeError("未检测到可用的GPU设备")
        requested = max(1, self.gpu_parallel_degree or 1)
        parallel = min(requested, available_gpus)
        if parallel < requested:
            print(f"⚠️ 请求的GPU并行度 {requested} 超出可用数量 {available_gpus}, 实际使用 {parallel}")
        if self._gpu_resources is None or len(self._gpu_resources) != parallel:
            self._gpu_resources = [faiss.StandardGpuResources() for _ in range(parallel)]
        device_ids = list(range(parallel))
        self._effective_gpu_parallel_degree = parallel
        return self._gpu_resources, device_ids

    def _ensure_faiss_index(self, vector_dim: int, total_vectors: int) -> None:
        if self.faiss_index is None:
            self.faiss_index = self._create_cpu_ivf_index(vector_dim, total_vectors)
            self._faiss_on_gpu = False

    def _train_index_if_needed(self, vectors: np.ndarray) -> None:
        if hasattr(self.faiss_index, "is_trained") and not self.faiss_index.is_trained:
            print("正在训练Faiss索引...")
            self.faiss_index.train(vectors)

    def _ensure_nprobe(self) -> None:
        if self.faiss_index is None:
            return
        if self.index_nprobe is None:
            source_nlist = None
            if hasattr(self.faiss_index, "nlist"):
                source_nlist = getattr(self.faiss_index, "nlist", None)
            if source_nlist is None and self.index_nlist is not None:
                source_nlist = self.index_nlist
            if source_nlist is not None:
                self.index_nprobe = min(16, max(1, int(source_nlist)))
            else:
                self.index_nprobe = 1
        target_nprobe = int(self.index_nprobe)
        if hasattr(self.faiss_index, "nlist"):
            target_nprobe = min(target_nprobe, getattr(self.faiss_index, "nlist", target_nprobe))
        if hasattr(self.faiss_index, "nprobe"):
            self.faiss_index.nprobe = target_nprobe
        elif hasattr(faiss, "ParameterSpace"):
            try:
                faiss.ParameterSpace().set_index_parameter(self.faiss_index, "nprobe", target_nprobe)
            except Exception as exc:
                print(f"⚠️ 无法在当前索引上设置nprobe参数({exc})")

    def _finalize_faiss_index(self) -> None:
        if self.faiss_index is None:
            return
        self._ensure_nprobe()
        if self.use_gpu_index and not self._faiss_on_gpu:
            try:
                resources, device_ids = self._prepare_gpu_resources()
                active_device_ids = device_ids
                if len(device_ids) == 1:
                    self.faiss_index = faiss.index_cpu_to_gpu(resources[0], device_ids[0], self.faiss_index)
                else:
                    if hasattr(faiss, "index_cpu_to_gpu_multiple_py"):
                        cloner_opts = faiss.GpuMultipleClonerOptions() if hasattr(faiss, "GpuMultipleClonerOptions") else None
                        self.faiss_index = faiss.index_cpu_to_gpu_multiple_py(resources, self.faiss_index, cloner_opts)
                    else:
                        print("⚠️ 当前Faiss版本不支持多GPU索引克隆, 将退回单GPU模式")
                        self.faiss_index = faiss.index_cpu_to_gpu(resources[0], device_ids[0], self.faiss_index)
                        active_device_ids = [device_ids[0]]
                self._faiss_on_gpu = True
                self._gpu_device_ids = active_device_ids
                self._effective_gpu_parallel_degree = len(active_device_ids)
                if hasattr(self.faiss_index, "nlist"):
                    self.index_nlist = self.faiss_index.nlist
                self._ensure_nprobe()
                if len(active_device_ids) > 1:
                    print(f"✓ 索引已迁移到GPU, 并行度: {len(active_device_ids)}")
                else:
                    print("✓ 索引已迁移到GPU")
            except Exception as e:
                print(f"⚠️ 索引迁移到GPU失败({e}),继续使用CPU索引")
                self._faiss_on_gpu = False
                self._gpu_device_ids = None
                self._effective_gpu_parallel_degree = 0
                if hasattr(self.faiss_index, "nlist"):
                    self.index_nlist = self.faiss_index.nlist
                self._ensure_nprobe()
        elif not self.use_gpu_index:
            self._faiss_on_gpu = False
            self._ensure_nprobe()

    def _cpu_index_for_persistence(self) -> Optional["faiss.Index"]:
        if self.faiss_index is None:
            return None
        if self._faiss_on_gpu:
            return faiss.index_gpu_to_cpu(self.faiss_index)
        return self.faiss_index

    def build_index_streaming(self, file_path: str, index_path: str,
                             batch_size: int = 64, chunk_size: int = 10000,
                             max_chunks: Optional[int] = None, num_workers: int = 1,
                             resume: bool = True):
        """
        流式构建Faiss索引，支持超大规模知识库
        
        Args:
            file_path: 知识库文件路径 (仅支持jsonl格式)
            index_path: 索引保存目录
            batch_size: embedding生成的批量大小
            chunk_size: 每次处理的文本块数量（控制内存使用）
            max_chunks: 最大处理的文本块数量
            num_workers: 并行处理的worker数量
            resume: 是否从上次中断处继续
        """
        if not file_path.endswith('.jsonl'):
            raise ValueError("流式构建仅支持JSONL格式文件")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 自动确定worker数量
        num_workers = self._resolve_num_workers(num_workers)
        
        os.makedirs(index_path, exist_ok=True)
        
        # 检查checkpoint
        checkpoint_file = os.path.join(index_path, "checkpoint.json")
        start_line = 0
        processed_chunks = 0
        
        if resume and os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                start_line = checkpoint.get("processed_lines", 0)
                processed_chunks = checkpoint.get("processed_chunks", 0)
            print(f"💾 从checkpoint恢复: 已处理 {processed_chunks:,} 个文本块")
            
            # 加载已有的chunks和Faiss索引
            self._load_partial_index(index_path)
        
        if self.faiss_index is None:
            print("将在首次批处理时自动初始化Faiss IndexIVFFlat")
        
        print(f"\n{'='*60}")
        print(f"🌊 Faiss流式索引构建模式")
        print(f"{'='*60}")
        print(f"文件路径: {file_path}")
        print(f"索引路径: {index_path}")
        print(f"每批处理: {chunk_size:,} 个文本块")
        print(f"Batch大小: {batch_size}")
        print(f"Worker数: {num_workers}")
        print(f"{'='*60}\n")
        
        total_start_time = time.time()
        batch_chunks = []
        batch_texts = []
        current_line = 0
        
        # 打开文件并逐行读取
        with open(file_path, 'r', encoding='utf-8') as f:
            # 跳过已处理的行
            if start_line > 0:
                print(f"⏭️  跳过已处理的 {start_line:,} 行...")
                for _ in range(start_line):
                    next(f)
                current_line = start_line
            
            pbar = tqdm(desc="处理文本块", unit="块", initial=processed_chunks)
            
            for line in f:
                current_line += 1
                
                if not line.strip():
                    continue
                
                try:
                    sample = json.loads(line)
                    
                    if not (isinstance(sample, dict) and "text" in sample and 
                           isinstance(sample["text"], str) and sample["text"]):
                        continue
                    
                    title = sample.get("title", "")
                    text = sample.get("text", "")
                    combined_text = f"{title}\n\n{text}" if title else text
                    
                    batch_chunks.append(sample)
                    batch_texts.append(combined_text)
                    
                    if len(batch_chunks) >= chunk_size or \
                       (max_chunks and processed_chunks + len(batch_chunks) >= max_chunks):
                        
                        # 处理并添加到Faiss索引
                        self._process_and_add_to_faiss(
                            batch_chunks, batch_texts,
                            index_path, batch_size, num_workers
                        )
                        
                        processed_chunks += len(batch_chunks)
                        pbar.update(len(batch_chunks))
                        
                        # 保存checkpoint
                        self._save_checkpoint(checkpoint_file, current_line, processed_chunks)
                        
                        batch_chunks = []
                        batch_texts = []
                        gc.collect()
                        
                        if max_chunks and processed_chunks >= max_chunks:
                            print(f"\n⚠️  已达到最大文本块数限制: {max_chunks:,}")
                            break
                
                except json.JSONDecodeError as e:
                    print(f"⚠️  跳过无效JSON (行{current_line}): {e}")
                    continue
            
            # 处理剩余批次
            if batch_chunks:
                self._process_and_add_to_faiss(
                    batch_chunks, batch_texts,
                    index_path, batch_size, num_workers
                )
                processed_chunks += len(batch_chunks)
                pbar.update(len(batch_chunks))
                self._save_checkpoint(checkpoint_file, current_line, processed_chunks)
            
            pbar.close()
        
        # 最终保存
        print(f"\n{'='*60}")
        print(f"💾 保存最终索引")
        print(f"{'='*60}")
        self._finalize_faiss_index()
        self.save_index(index_path)
        
        # 删除checkpoint
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        
        total_time = time.time() - total_start_time
        print(f"\n{'='*60}")
        print(f"✅ Faiss流式索引构建完成!")
        print(f"{'='*60}")
        print(f"总文本块数: {processed_chunks:,}")
        print(f"Faiss索引包含: {self.faiss_index.ntotal:,} 个向量")
        print(f"总耗时: {total_time:.2f}秒")
        print(f"平均速度: {processed_chunks/total_time:.1f} 块/秒")
        print(f"索引路径: {index_path}")
        print(f"{'='*60}\n")
    
    def _process_and_add_to_faiss(self, chunks: List[Dict], texts: List[str],
                                  index_path: str, batch_size: int, num_workers: int):
        """处理批次并直接添加到Faiss索引"""
        # 生成embeddings
        vectors = self._get_embeddings_batch(
            texts,
            batch_size=batch_size,
            show_progress=False,
            num_workers=num_workers
        )
        
        # 归一化向量
        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        faiss.normalize_L2(vectors)
        
        total_vectors = len(self.chunks) + vectors.shape[0]
        self._ensure_faiss_index(vectors.shape[1], total_vectors)
        self._train_index_if_needed(vectors)
        self.faiss_index.add(vectors)
        self._ensure_nprobe()
        
        # 保存chunks（追加模式）
        self.chunks.extend(chunks)
        
        # 将vectors添加到内存中（用于最后保存）
        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])
        
        # 定期保存中间结果
        if len(self.chunks) % 50000 < len(chunks):  # 每5万条保存一次
            print(f"  💾 保存中间结果 ({len(self.chunks):,} 个文本块)...")
            self._save_partial_index(index_path)
    
    def _save_partial_index(self, index_path: str):
        """保存部分索引（中间checkpoint）"""
        partial_dir = os.path.join(index_path, "partial")
        os.makedirs(partial_dir, exist_ok=True)
        
        # 保存chunks
        chunks_file = os.path.join(partial_dir, "chunks.json")
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, ensure_ascii=False)
        
        # 保存Faiss索引
        faiss_file = os.path.join(partial_dir, "faiss.index")
        index_to_save = self._cpu_index_for_persistence()
        if index_to_save is not None:
            faiss.write_index(index_to_save, faiss_file)
        
        # 保存numpy vectors
        if self.vectors is not None:
            vectors_file = os.path.join(partial_dir, "vectors.npy")
            np.save(vectors_file, self.vectors)
    
    def _load_partial_index(self, index_path: str):
        """加载部分索引（用于resume）"""
        partial_dir = os.path.join(index_path, "partial")
        
        if not os.path.exists(partial_dir):
            return
        
        # 加载chunks
        chunks_file = os.path.join(partial_dir, "chunks.json")
        if os.path.exists(chunks_file):
            with open(chunks_file, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
            print(f"✓ 加载已有chunks: {len(self.chunks):,} 个")
        
        # 加载Faiss索引
        faiss_file = os.path.join(partial_dir, "faiss.index")
        if os.path.exists(faiss_file):
            self.faiss_index = faiss.read_index(faiss_file)
            if hasattr(self.faiss_index, "nlist"):
                self.index_nlist = self.faiss_index.nlist
                self.index_nprobe = min(16, max(1, self.faiss_index.nlist))
                self._ensure_nprobe()
            self._faiss_on_gpu = False
            print(f"✓ 加载已有Faiss索引: {self.faiss_index.ntotal:,} 个向量")
        
        # 加载vectors
        vectors_file = os.path.join(partial_dir, "vectors.npy")
        if os.path.exists(vectors_file):
            self.vectors = np.load(vectors_file)
            print(f"✓ 加载已有vectors: {self.vectors.shape}")

    def build_index(self, file_path: str, batch_size: int = 64,
                   max_chunks: Optional[int] = None, num_workers: int = 1):
        """
        构建Faiss索引（常规模式）
        
        Args:
            file_path: 知识库文件路径
            batch_size: embedding生成批量大小
            max_chunks: 最大处理的文本块数量
            num_workers: 并行处理的worker数量
        
        提示: 对于超大规模知识库，建议使用 build_index_streaming() 方法
        """
        # 调用父类方法生成embeddings
        super().build_index(file_path, batch_size, max_chunks, num_workers)
        
        # 构建Faiss索引
        print(f"\n{'='*60}")
        print(f"🚀 步骤 5/5: 构建Faiss索引")
        print(f"{'='*60}")
        
        try:
            start_time = time.time()
            if self.vectors is None:
                raise RuntimeError("归一化向量缺失，无法构建Faiss索引")
            vectors = np.ascontiguousarray(self.vectors.astype(np.float32))
            d = vectors.shape[1]
            total_vectors = vectors.shape[0]
            
            print(f"向量维度: {d}")
            print(f"向量数量: {len(vectors):,}")
            
            # 构建Faiss IndexIVFFlat
            self.faiss_index = None
            self._faiss_on_gpu = False
            self._ensure_faiss_index(d, total_vectors)
            self._train_index_if_needed(vectors)
            
            print("正在添加向量到Faiss索引...")
            self.faiss_index.add(vectors)
            self._ensure_nprobe()
            self._finalize_faiss_index()
            
            elapsed = time.time() - start_time
            print(f"✓ Faiss索引构建完成!")
            print(f"  - 包含向量数: {self.faiss_index.ntotal:,}")
            if self.index_nlist is not None:
                print(f"  - nlist: {self.index_nlist}")
            if self.index_nprobe is not None:
                print(f"  - nprobe: {self.index_nprobe}")
            print(f"  - 构建耗时: {elapsed:.2f}秒")
            
            # 清理内存
            del vectors
            gc.collect()
            
        except Exception as e:
            print(f"✗ 构建Faiss索引时出错: {str(e)}")
            raise

    def save_index(self, index_path: str):
        """
        保存Faiss索引
        
        Args:
            index_path: 保存索引的目录路径
        """
        if self.faiss_index is None or not self.chunks:
            raise ValueError("索引为空,无法保存")

        os.makedirs(index_path, exist_ok=True)
        
        # 保存文本块
        chunks_file = os.path.join(index_path, "chunks.json")
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        
        # 保存Faiss索引
        index_file = os.path.join(index_path, "faiss.index")
        self._ensure_nprobe()
        index_to_save = self._cpu_index_for_persistence()
        if index_to_save is None:
            raise RuntimeError("Faiss索引未初始化，无法保存")
        faiss.write_index(index_to_save, index_file)
        
        # 保存元数据
        metadata = {
            "model_name": self.model_name,
            "device": self.device,
            "max_seq_length": self.max_seq_length,
            "num_chunks": len(self.chunks),
            "vector_dim": self.faiss_index.d if hasattr(self.faiss_index, 'd') else None,
            "use_gpu_index": self.use_gpu_index,
            "index_type": "faiss.IndexIVFFlat",
            "nlist": self.index_nlist,
            "nprobe": self.index_nprobe,
            "embedding_devices": self.embedding_devices,
        }
        metadata_file = os.path.join(index_path, "metadata.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Faiss索引已保存到: {index_path}")

    @classmethod
    def load_index(cls, index_path: str, model_name: Optional[str] = None,
                   device: str = "cuda", max_seq_length: int = 512,
                   use_gpu_index: bool = False, memory_map: bool = True,
                   read_only_index: bool = True,
                   embedding_devices: Optional[Sequence[str]] = None, **kwargs):
        """
        加载Faiss索引
        
        Args:
            index_path: 索引目录路径
            model_name: 模型名称
            device: 运行设备
            max_seq_length: 最大序列长度
            use_gpu_index: 是否使用GPU索引
            memory_map: 是否开启磁盘内存映射以降低常驻内存
            read_only_index: 是否以只读方式打开索引（配合内存映射）
        
        Returns:
            加载的RAGIndexLocal_faiss实例
        """
        chunks_file = os.path.join(index_path, "chunks.json")
        index_file = os.path.join(index_path, "faiss.index")
        metadata_file = os.path.join(index_path, "metadata.json")

        # 检查文件存在性
        if not os.path.exists(chunks_file):
            raise FileNotFoundError(f"未找到chunks文件: {chunks_file}")
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"未找到Faiss索引文件: {index_file}")
        
        # 读取元数据
        metadata = None
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            if model_name is None:
                model_name = metadata.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
            if embedding_devices is None:
                embedding_devices = metadata.get("embedding_devices")
            print(f"✓ 从元数据读取配置: {metadata}")
        else:
            if model_name is None:
                model_name = "sentence-transformers/all-MiniLM-L6-v2"
            print("⚠️ 未找到元数据文件,使用默认配置")

        kw_embedding = kwargs.pop("embedding_devices", None)
        if embedding_devices is None and kw_embedding is not None:
            embedding_devices = kw_embedding

        # 创建实例
        instance = cls(
            model_name=model_name,
            device=device,
            max_seq_length=max_seq_length,
            use_gpu_index=use_gpu_index,
             embedding_devices=embedding_devices,
            **kwargs
        )
        
        # 加载文本块
        chunks_jsonl = os.path.join(index_path, "chunks.jsonl")
        chunks_offsets = os.path.join(index_path, "chunks.offsets")

        if os.path.exists(chunks_jsonl) and os.path.exists(chunks_offsets):
             print(f"✓ 检测到优化存储: 使用 DiskBasedChunks (按需读取 {chunks_jsonl})")
             instance.chunks = DiskBasedChunks(chunks_jsonl, chunks_offsets)
        elif os.path.exists(chunks_file):
             with open(chunks_file, 'r', encoding='utf-8') as f:
                instance.chunks = json.load(f)
        
        # 加载Faiss索引
        io_flags = 0
        mmap_supported = hasattr(faiss, "IO_FLAG_MMAP")
        readonly_supported = hasattr(faiss, "IO_FLAG_READ_ONLY")
        attempted_mmap = False
        if memory_map and mmap_supported:
            io_flags |= faiss.IO_FLAG_MMAP
            attempted_mmap = True
        if read_only_index and readonly_supported:
            io_flags |= faiss.IO_FLAG_READ_ONLY
        load_errors = []
        cpu_index = None
        if io_flags:
            try:
                cpu_index = faiss.read_index(index_file, io_flags)
                if attempted_mmap:
                    print("✓ 以内存映射方式加载Faiss索引，减少常驻内存占用")
            except Exception as exc:  # pylint: disable=broad-except
                load_errors.append(exc)
                print(f"⚠️ 内存映射加载失败({exc})，将回退到常规加载")
        if cpu_index is None:
            try:
                cpu_index = faiss.read_index(index_file)
            except Exception as exc:
                load_errors.append(exc)
                raise RuntimeError(
                    f"无法加载Faiss索引: {index_file}，错误: {load_errors}"
                ) from exc

        instance.faiss_index = cpu_index
        instance._faiss_on_gpu = False
        if hasattr(cpu_index, "nlist"):
            instance.index_nlist = cpu_index.nlist
        metadata_nprobe = metadata.get("nprobe") if metadata else None
        if metadata_nprobe is not None:
            instance.index_nprobe = int(metadata_nprobe)
        elif instance.index_nlist is not None:
            instance.index_nprobe = min(16, max(1, instance.index_nlist))
        else:
            instance.index_nprobe = 1
        instance._ensure_nprobe()
        
        instance._finalize_faiss_index()

        print(f"✓ 成功加载Faiss索引: {len(instance.chunks)} 个文本块")
        
        return instance

    def query(self, query: str, top_k: int = 3) -> str:
        """
        使用Faiss查询索引
        
        Args:
            query: 查询文本
            top_k: 返回前k个结果
        
        Returns:
            检索到的上下文文本
        """
        if self.faiss_index is None:
            raise RuntimeError("Faiss索引未构建或加载,无法查询")
        
        if not query:
            raise ValueError("查询文本不能为空")


        try:
            # 生成查询向量
            query_vector = self.model.encode(
                [query],
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=False
            )[0].astype(np.float32)
            
            # 归一化
            query_vector = query_vector.reshape(1, -1)
            faiss.normalize_L2(query_vector)
            
            self._ensure_nprobe()
            
            # 搜索
            distances, indices = self.faiss_index.search(query_vector, top_k)
            indices = indices[0]
            
            # 获取结果，确保索引有效
            retrieved_chunks: List[Dict[str, Any]] = []
            for raw_idx in indices:
                idx = int(raw_idx)
                if idx < 0:
                    continue
                if idx >= len(self.chunks):
                    print(f"⚠️ 查询结果索引越界: {idx} (chunks={len(self.chunks)})，已忽略")
                    continue
                retrieved_chunks.append(self.chunks[idx])

            if not retrieved_chunks:
                return "[查询错误] 未检索到有效的文本块"

            formatted_chunks = [self._format_chunk_for_output(chunk) for chunk in retrieved_chunks]
            context = "\n---\n".join(formatted_chunks)
            
            return f"### Retrieved Context:\n{context}"
        except Exception as e:
            return f"[查询错误] {str(e)}"


class RAGIndexLocal_faiss_compact(RAGIndexLocal_faiss):
    """
    基于IVFPQ压缩的Faiss索引实现, 目标是在大规模知识库场景下显著降低索引体积。
    
    特性:
        - 使用 IVF + PQ (Product Quantization) 组合, 默认约 24 字节/向量
        - 训练阶段自动缓冲数据, 确保PQ训练稳定, 并支持小样本自动回退
        - 可选 FP16 内存缓存减少构建期内存消耗
        - 兼容原有的 build_index / build_index_streaming 接口和 load/save 协议
    """

    def __init__(self,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cuda",
                 max_seq_length: int = 512,
                 use_gpu_index: bool = False,
                 gpu_parallel_degree: Optional[int] = None,
                 embedding_devices: Optional[Sequence[str]] = None,
                 pq_m: Optional[int] = None,
                 pq_bits: int = 8,
                 target_bytes_per_vector: Optional[int] = 24,
                 training_samples: int = 262_144,
                 store_embeddings_fp16: bool = True) -> None:
        """
        Args:
            pq_m: Product Quantization 分块数, None 时根据 target_bytes_per_vector 自动推导
            pq_bits: 每个子向量的码字位数 (默认8, 支持4/5/6/7/8)
            target_bytes_per_vector: 希望的压缩后字节数, 将近似映射为 pq_m
            training_samples: 用于IVF+PQ训练的最大样本量
            store_embeddings_fp16: 构建流程中是否以FP16缓存向量以降低内存
        """
        super().__init__(
            model_name=model_name,
            device=device,
            max_seq_length=max_seq_length,
            use_gpu_index=use_gpu_index,
            gpu_parallel_degree=gpu_parallel_degree,
            embedding_devices=embedding_devices,
        )
        self.pq_m = int(pq_m) if pq_m is not None else None
        self.pq_bits = max(4, min(8, int(pq_bits)))
        self.target_bytes_per_vector = target_bytes_per_vector
        self.training_samples = max(2048, int(training_samples))
        self.store_embeddings_fp16 = store_embeddings_fp16

        self._effective_pq_m: Optional[int] = self.pq_m
        self._min_training_vectors: Optional[int] = None
        self._pending_training_vectors: List[np.ndarray] = []
        self._pending_training_chunks: List[Dict[str, Any]] = []
        self._pending_training_count: int = 0
        self._fallback_to_flat: bool = False

    @staticmethod
    def _divisors(value: int) -> List[int]:
        divs: set[int] = set()
        upper = int(math.sqrt(value)) + 1
        for factor in range(1, upper):
            if value % factor == 0:
                divs.add(factor)
                divs.add(value // factor)
        return sorted(divs)

    def _resolve_pq_m(self, vector_dim: int) -> int:
        if self.pq_m is not None:
            if vector_dim % self.pq_m != 0:
                raise ValueError(
                    f"pq_m={self.pq_m} 与向量维度 {vector_dim} 不整除, 请调整 pq_m 或使用 target_bytes_per_vector 自动推导"
                )
            return self.pq_m

        divisors = [d for d in self._divisors(vector_dim) if d > 1]
        if not divisors:
            return 1
        target = max(8, int(self.target_bytes_per_vector)) if self.target_bytes_per_vector else 8
        candidate = min(divisors, key=lambda d: (abs(d - target), d))
        candidate = max(8, candidate)

        # 确保能被维度整除
        while vector_dim % candidate != 0 and candidate < vector_dim:
            candidate += 1
        if vector_dim % candidate != 0:
            # 兜底选择最大的可行因子
            candidate = max(d for d in divisors if vector_dim % d == 0)
        return max(4, min(candidate, vector_dim))

    def _compute_min_training(self, vector_dim: int) -> int:
        if self._effective_pq_m is None:
            self._effective_pq_m = self._resolve_pq_m(vector_dim)
        sub_vector_train = self._effective_pq_m * (1 << self.pq_bits)
        coarse_train = max(1024, (self.index_nlist or 1) * 16)
        return max(2048, sub_vector_train, coarse_train)

    def _create_cpu_ivf_index(self, vector_dim: int, total_vectors: int) -> "faiss.Index":
        pq_m = self._resolve_pq_m(vector_dim)
        nlist, nprobe = _suggest_ivf_params(max(total_vectors, pq_m * 64))
        quantizer = faiss.IndexFlatIP(vector_dim)
        try:
            index = faiss.IndexIVFPQ(quantizer, vector_dim, nlist, pq_m, self.pq_bits)
            print(
                f"✓ 使用IVFPQ索引 (维度={vector_dim}, nlist={nlist}, m={pq_m}, bits={self.pq_bits}) "
                f"≈ {pq_m} 字节/向量"
            )
        except Exception as exc:
            raise RuntimeError(f"初始化IVFPQ索引失败: {exc}") from exc

        index.nprobe = min(nprobe, nlist)
        self.index_nlist = nlist
        self.index_nprobe = index.nprobe
        self._effective_pq_m = pq_m
        self._min_training_vectors = self._compute_min_training(vector_dim)
        return index

    def _store_embeddings(self, vectors: np.ndarray) -> None:
        super()._store_embeddings(vectors)
        if self.store_embeddings_fp16 and self.vectors is not None:
            self.vectors = np.ascontiguousarray(self.vectors.astype(np.float16))
            print("✓ 向量缓存已转换为FP16以降低内存占用")

    def _append_training_buffer(self, vectors: np.ndarray, chunks: List[Dict[str, Any]]) -> None:
        self._pending_training_vectors.append(vectors)
        self._pending_training_chunks.extend(chunks)
        self._pending_training_count += vectors.shape[0]

    def _clear_training_buffer(self) -> None:
        self._pending_training_vectors = []
        self._pending_training_chunks = []
        self._pending_training_count = 0

    def _attempt_training(self, force: bool = False) -> None:
        if self.faiss_index is None:
            return
        if not hasattr(self.faiss_index, "is_trained"):
            return

        if self.faiss_index.is_trained:
            if self._pending_training_vectors:
                pending = np.vstack(self._pending_training_vectors)
                self.faiss_index.add(pending)
                self._ensure_nprobe()
                self.chunks.extend(self._pending_training_chunks)
                self._clear_training_buffer()
            return

        if not self._pending_training_vectors:
            return

        sample = np.vstack(self._pending_training_vectors)
        min_needed = self._min_training_vectors or self._compute_min_training(sample.shape[1])
        if not force and sample.shape[0] < min_needed:
            return

        train_limit = min(sample.shape[0], self.training_samples)
        train_data = sample[:train_limit]
        try:
            self.faiss_index.train(train_data)
            print(f"✓ IVFPQ索引训练完成, 使用 {train_limit:,} 条向量作为训练样本")
        except Exception as exc:
            print(f"⚠️ IVFPQ训练失败({exc}), 回退到 IndexIVFFlat")
            self._fallback_to_ivfflat(sample)
            return

        self.faiss_index.add(sample)
        self._ensure_nprobe()
        self.chunks.extend(self._pending_training_chunks)
        self._clear_training_buffer()

    def _fallback_to_ivfflat(self, sample: np.ndarray) -> None:
        backup_index = super()._create_cpu_ivf_index(sample.shape[1], sample.shape[0])
        if hasattr(backup_index, "is_trained") and not backup_index.is_trained:
            backup_index.train(sample)
        backup_index.add(sample)
        self.faiss_index = backup_index
        self._fallback_to_flat = True
        self._ensure_nprobe()
        self._clear_training_buffer()
        print("✓ 已回退至 IndexIVFFlat, 仍可继续构建但索引体积会增大")

    def _process_and_add_to_faiss(self, chunks: List[Dict], texts: List[str],
                                  index_path: str, batch_size: int, num_workers: int):
        vectors = self._get_embeddings_batch(
            texts,
            batch_size=batch_size,
            show_progress=False,
            num_workers=num_workers
        )
        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        faiss.normalize_L2(vectors)

        total_vectors = len(self.chunks) + self._pending_training_count + vectors.shape[0]
        self._ensure_faiss_index(vectors.shape[1], total_vectors)

        if hasattr(self.faiss_index, "is_trained") and not self.faiss_index.is_trained and not self._fallback_to_flat:
            self._append_training_buffer(vectors, chunks)
            self._attempt_training(force=False)
            return

        # 索引已训练或已回退为IVFFlat
        self.faiss_index.add(vectors)
        self._ensure_nprobe()
        self.chunks.extend(chunks)

        # streaming 下仍然支持checkpoint
        if len(self.chunks) % 50000 < len(chunks):
            print(f"  💾 保存中间结果 ({len(self.chunks):,} 个文本块)...")
            self._save_partial_index(index_path)

    def _save_partial_index(self, index_path: str):
        partial_dir = os.path.join(index_path, "partial")
        os.makedirs(partial_dir, exist_ok=True)

        chunks_file = os.path.join(partial_dir, "chunks.json")
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, ensure_ascii=False)

        faiss_file = os.path.join(partial_dir, "faiss.index")
        index_to_save = self._cpu_index_for_persistence()
        if index_to_save is not None:
            faiss.write_index(index_to_save, faiss_file)

    def _load_partial_index(self, index_path: str):
        partial_dir = os.path.join(index_path, "partial")
        if not os.path.exists(partial_dir):
            return

        chunks_file = os.path.join(partial_dir, "chunks.json")
        if os.path.exists(chunks_file):
            with open(chunks_file, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)

        faiss_file = os.path.join(partial_dir, "faiss.index")
        if os.path.exists(faiss_file):
            self.faiss_index = faiss.read_index(faiss_file)
            self._faiss_on_gpu = False
            if hasattr(self.faiss_index, "nlist"):
                self.index_nlist = self.faiss_index.nlist
            if hasattr(self.faiss_index, "nprobe"):
                self.index_nprobe = self.faiss_index.nprobe
            if hasattr(self.faiss_index, "pq"):
                self._effective_pq_m = getattr(self.faiss_index.pq, "M", None)
            self._ensure_nprobe()

    def _finalize_faiss_index(self) -> None:
        self._attempt_training(force=True)
        if hasattr(self.faiss_index, "is_trained") and not self.faiss_index.is_trained:
            raise RuntimeError("Faiss索引尚未完成训练，无法最终化")
        super()._finalize_faiss_index()
        # 构建完毕可释放缓存向量
        self.vectors = None

    def build_index(self, file_path: str, batch_size: int = 64,
                    max_chunks: Optional[int] = None, num_workers: int = 1):
        super().build_index(file_path, batch_size, max_chunks, num_workers)
        self.vectors = None
        gc.collect()

    def save_index(self, index_path: str):
        if self.faiss_index is None or not self.chunks:
            raise ValueError("索引为空,无法保存")

        os.makedirs(index_path, exist_ok=True)

        chunks_file = os.path.join(index_path, "chunks.json")
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

        self._ensure_nprobe()
        index_to_save = self._cpu_index_for_persistence()
        if index_to_save is None:
            raise RuntimeError("Faiss索引未初始化，无法保存")

        index_file = os.path.join(index_path, "faiss.index")
        faiss.write_index(index_to_save, index_file)

        metadata = {
            "model_name": self.model_name,
            "device": self.device,
            "max_seq_length": self.max_seq_length,
            "num_chunks": len(self.chunks),
            "vector_dim": self.faiss_index.d if hasattr(self.faiss_index, 'd') else None,
            "use_gpu_index": self.use_gpu_index,
            "index_type": "faiss.IndexIVFPQ" if not self._fallback_to_flat else "faiss.IndexIVFFlat",
            "nlist": self.index_nlist,
            "nprobe": self.index_nprobe,
            "pq_m": self._effective_pq_m,
            "pq_bits": self.pq_bits,
            "target_bytes_per_vector": self.target_bytes_per_vector,
            "store_embeddings_fp16": self.store_embeddings_fp16,
            "training_samples": self.training_samples,
            "fallback_to_flat": self._fallback_to_flat,
            "embedding_devices": self.embedding_devices,
        }
        metadata_file = os.path.join(index_path, "metadata.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"✓ 压缩Faiss索引已保存到: {index_path}")

    @classmethod
    def load_index(cls, index_path: str, model_name: Optional[str] = None,
                   device: str = "cuda", max_seq_length: int = 512,
                   use_gpu_index: bool = False, memory_map: bool = True,
                   read_only_index: bool = True,
                   embedding_devices: Optional[Sequence[str]] = None, **kwargs):
        chunks_file = os.path.join(index_path, "chunks.json")
        index_file = os.path.join(index_path, "faiss.index")
        metadata_file = os.path.join(index_path, "metadata.json")

        if not os.path.exists(chunks_file):
            raise FileNotFoundError(f"未找到chunks文件: {chunks_file}")
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"未找到Faiss索引文件: {index_file}")

        metadata: Optional[Dict[str, Any]] = None
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            if model_name is None:
                model_name = metadata.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
            if embedding_devices is None:
                embedding_devices = metadata.get("embedding_devices")
            print(f"✓ 从元数据读取配置: {metadata}")
        else:
            if model_name is None:
                model_name = "sentence-transformers/all-MiniLM-L6-v2"
            print("⚠️ 未找到元数据文件,使用默认配置")

        kw_embedding = kwargs.pop("embedding_devices", None)
        if embedding_devices is None and kw_embedding is not None:
            embedding_devices = kw_embedding

        pq_m = metadata.get("pq_m") if metadata else None
        pq_bits = metadata.get("pq_bits", 8) if metadata else 8
        target_bytes = metadata.get("target_bytes_per_vector") if metadata else kwargs.get("target_bytes_per_vector")
        store_fp16 = metadata.get("store_embeddings_fp16", True) if metadata else kwargs.get("store_embeddings_fp16", True)
        training_samples = metadata.get("training_samples", kwargs.get("training_samples", 262_144)) if metadata else kwargs.get("training_samples", 262_144)

        instance = cls(
            model_name=model_name,
            device=device,
            max_seq_length=max_seq_length,
            use_gpu_index=use_gpu_index,
            embedding_devices=embedding_devices,
            pq_m=pq_m,
            pq_bits=pq_bits,
            target_bytes_per_vector=target_bytes,
            training_samples=training_samples,
            store_embeddings_fp16=store_fp16,
            **{k: v for k, v in kwargs.items() if k not in {"target_bytes_per_vector", "store_embeddings_fp16", "training_samples"}}
        )

        # 加载文本块
        chunks_jsonl = os.path.join(index_path, "chunks.jsonl")
        chunks_offsets = os.path.join(index_path, "chunks.offsets")
        
        if os.path.exists(chunks_jsonl) and os.path.exists(chunks_offsets):
             print(f"✓ 检测到优化存储: 使用 DiskBasedChunks (按需读取 {chunks_jsonl})")
             instance.chunks = DiskBasedChunks(chunks_jsonl, chunks_offsets)
        elif os.path.exists(chunks_file):
             with open(chunks_file, 'r', encoding='utf-8') as f:
                instance.chunks = json.load(f)

        io_flags = 0
        mmap_supported = hasattr(faiss, "IO_FLAG_MMAP")
        readonly_supported = hasattr(faiss, "IO_FLAG_READ_ONLY")
        attempted_mmap = False
        if memory_map and mmap_supported:
            io_flags |= faiss.IO_FLAG_MMAP
            attempted_mmap = True
        if read_only_index and readonly_supported:
            io_flags |= faiss.IO_FLAG_READ_ONLY

        load_errors = []
        cpu_index = None
        if io_flags:
            try:
                cpu_index = faiss.read_index(index_file, io_flags)
                if attempted_mmap:
                    print("✓ 以内存映射方式加载压缩Faiss索引")
            except Exception as exc:  # pylint: disable=broad-except
                load_errors.append(exc)
                print(f"⚠️ 内存映射加载失败({exc})，将回退到常规加载")
        if cpu_index is None:
            try:
                cpu_index = faiss.read_index(index_file)
            except Exception as exc:
                load_errors.append(exc)
                raise RuntimeError(
                    f"无法加载Faiss索引: {index_file}，错误: {load_errors}"
                ) from exc

        instance.faiss_index = cpu_index
        instance._faiss_on_gpu = False
        if hasattr(cpu_index, "nlist"):
            instance.index_nlist = cpu_index.nlist
        if hasattr(cpu_index, "nprobe"):
            instance.index_nprobe = cpu_index.nprobe
        elif metadata and metadata.get("nprobe") is not None:
            instance.index_nprobe = int(metadata["nprobe"])
        elif instance.index_nlist is not None:
            instance.index_nprobe = min(16, max(1, instance.index_nlist))
        else:
            instance.index_nprobe = 1

        if hasattr(cpu_index, "pq"):
            instance._effective_pq_m = getattr(cpu_index.pq, "M", pq_m)
            instance.pq_bits = getattr(cpu_index.pq, "nbits", instance.pq_bits)

        instance._fallback_to_flat = bool(metadata.get("fallback_to_flat")) if metadata else False
        instance._min_training_vectors = instance._compute_min_training(cpu_index.d) if hasattr(cpu_index, "d") else None
        instance._ensure_nprobe()
        instance._finalize_faiss_index()

        print(f"✓ 成功加载压缩Faiss索引: {len(instance.chunks)} 个文本块")
        return instance


def get_rag_index_class(use_faiss: bool = False, use_compact: bool = False, use_hybrid: bool = False):
    """根据配置获取本地RAG索引类"""
    # 优先使用混合检索模式
    if use_hybrid:
        print("✅ 使用 HybridRAGIndex (支持 BM25 + E5 混合检索)")
        return HybridRAGIndex

    if not _SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError("sentence_transformers 未安装, 无法使用本地RAG索引")

    if use_compact:
        if not _FAISS_AVAILABLE:
            print("⚠️ Faiss 不可用,无法使用 Compact 索引, 回退到 RAGIndexLocal (Numpy 实现)")
            return RAGIndexLocal
        print("✅ 使用 RAGIndexLocal_faiss_compact (本地embedding + Faiss压缩索引)")
        return RAGIndexLocal_faiss_compact

    if use_faiss:
        if not _FAISS_AVAILABLE:
            print("⚠️ Faiss 不可用,回退到 RAGIndexLocal (Numpy 实现)")
            return RAGIndexLocal
        print("✅ 使用 RAGIndexLocal_faiss (本地embedding + Faiss加速)")
        return RAGIndexLocal_faiss

    print("✅ 使用 RAGIndexLocal (本地embedding)")
    return RAGIndexLocal


# ============================================================================
# DecEx-RAG Style Hybrid Index (BM25 + E5)
# ============================================================================

class DecExEncoder:
    """
    移植自 DecEx-RAG 的 Encoder，用于 E5/BGE 等模型的稠密检索
    """
    def __init__(self, model_name: str, model_path: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device

        # 延迟导入以避免非 E5 场景的开销
        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError("DecExEncoder 需要 transformers 库: pip install transformers")

        print(f"[DecExEncoder] 正在加载模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.model.eval()
        self.model.to(self.device)
        print(f"[DecExEncoder] 模型加载完成，设备: {self.device}")

    def encode(self, query_list: List[str], max_length: int = 512) -> np.ndarray:
        if isinstance(query_list, str):
            query_list = [query_list]

        # E5 特有的 Instruction
        if "e5" in self.model_name.lower():
            query_list = [f"query: {q}" for q in query_list]

        inputs = self.tokenizer(
            query_list,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output = self.model(**inputs)
            # Mean Pooling
            attention_mask = inputs['attention_mask']
            last_hidden = output.last_hidden_state.masked_fill(
                ~attention_mask[..., None].bool(), 0.0
            )
            embeddings = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)

        return embeddings.cpu().numpy().astype(np.float32)


class BM25RAGIndex(BaseRAGIndex):
    """
    稀疏检索后端：封装 Pyserini BM25
    """
    def __init__(self, index_path: str, device: str = "cpu", **kwargs):
        self.index_path = index_path
        self.device = device

        try:
            from pyserini.search.lucene import LuceneSearcher
        except ImportError:
            raise ImportError("BM25RAGIndex 需要 pyserini 和 Java 环境: pip install pyserini")

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"BM25 索引路径不存在: {index_path}")

        print(f"[BM25] 正在加载索引: {index_path}")
        self.searcher = LuceneSearcher(self.index_path)
        print("[BM25] 索引加载完成")

    def query(self, query: str, top_k: int = 5, **kwargs) -> str:
        """执行 BM25 检索"""
        try:
            hits = self.searcher.search(query, k=top_k)
            results = []
            for hit in hits:
                raw_doc = self.searcher.doc(hit.docid).raw()
                content_json = json.loads(raw_doc)
                # 兼容 DecEx-RAG 语料格式
                text = content_json.get('contents') or content_json.get('text', '')
                title = content_json.get('title', '')
                results.append(f"[{title}]\n{text}" if title else text)

            if not results:
                return "[查询结果] 未找到相关文档"

            return "\n---\n".join(results)
        except Exception as e:
            return f"[BM25 Error] {str(e)}"

    # 桩方法（BM25 索引由外部工具构建）
    def build_index(self, *args, **kwargs):
        raise NotImplementedError("BM25 索引需要使用 pyserini 工具离线构建")

    def save_index(self, *args, **kwargs):
        pass

    def _store_embeddings(self, *args, **kwargs):
        pass

    def _finalize_embeddings(self, *args, **kwargs):
        pass

    @classmethod
    def load_index(cls, index_path: str, **kwargs) -> "BM25RAGIndex":
        return cls(index_path, **kwargs)


class DenseE5RAGIndex(BaseRAGIndex):
    """
    密集检索后端：封装 Faiss + E5 Encoder
    """
    def __init__(self, index_path: str, model_name: str, corpus_path: str,
                 device: str = "cuda", **kwargs):
        self.device = device
        self.model_name = model_name
        self.corpus_path = corpus_path

        if not _FAISS_AVAILABLE:
            raise ImportError("DenseE5RAGIndex 需要 faiss: pip install faiss-gpu 或 faiss-cpu")

        # 1. 加载 Faiss 索引
        print(f"[E5] 正在加载 Faiss 索引: {index_path}")
        self.index = faiss.read_index(index_path)

        # 尝试迁移到 GPU
        if "cuda" in device and hasattr(faiss, "StandardGpuResources"):
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                print("[E5] Faiss 索引已迁移到 GPU")
            except Exception as e:
                print(f"[E5] GPU 迁移失败，使用 CPU: {e}")

        # 2. 加载 Encoder
        print(f"[E5] 正在初始化 Encoder: {model_name}")
        self.encoder = DecExEncoder(model_name, model_path=model_name, device=device)

        # 3. 加载语料库 (支持 JSONL)
        print(f"[E5] 正在加载语料库: {corpus_path}")
        self.corpus = []
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="加载语料"):
                if line.strip():
                    self.corpus.append(json.loads(line))
        print(f"[E5] 语料库加载完成，共 {len(self.corpus):,} 条")

    def query(self, query: str, top_k: int = 5, **kwargs) -> str:
        """执行稠密检索"""
        try:
            query_emb = self.encoder.encode([query])
            scores, idxs = self.index.search(query_emb, k=top_k)

            results = []
            for idx in idxs[0]:
                if idx == -1 or idx >= len(self.corpus):
                    continue
                doc = self.corpus[idx]
                text = doc.get('contents') or doc.get('text', '')
                title = doc.get('title', '')
                results.append(f"[{title}]\n{text}" if title else text)

            if not results:
                return "[查询结果] 未找到相关文档"

            return "\n---\n".join(results)
        except Exception as e:
            return f"[E5 Error] {str(e)}"

    # 桩方法
    def build_index(self, *args, **kwargs):
        raise NotImplementedError("E5 索引需要使用专用脚本离线构建")

    def save_index(self, *args, **kwargs):
        pass

    def _store_embeddings(self, *args, **kwargs):
        pass

    def _finalize_embeddings(self, *args, **kwargs):
        pass

    @classmethod
    def load_index(cls, index_path: str, model_name: str, corpus_path: str, **kwargs) -> "DenseE5RAGIndex":
        return cls(index_path, model_name, corpus_path, **kwargs)


class HybridRAGIndex(BaseRAGIndex):
    """
    混合检索索引：支持 BM25 (sparse) 和 E5 (dense) 双模式
    通过 search_type 参数在运行时选择检索方式
    """
    def __init__(
        self,
        bm25_index_path: Optional[str] = None,
        dense_index_path: Optional[str] = None,
        dense_model_name: str = "intfloat/e5-base-v2",
        corpus_path: Optional[str] = None,
        device: str = "cuda",
        **kwargs
    ):
        """
        Args:
            bm25_index_path: BM25 索引路径（可选）
            dense_index_path: Dense 索引路径（可选）
            dense_model_name: Dense 模型名称
            corpus_path: 语料库路径（Dense 索引必需）
            device: 运行设备
        """
        self.bm25_index: Optional[BM25RAGIndex] = None
        self.dense_index: Optional[DenseE5RAGIndex] = None
        self.device = device

        # 懒加载：仅在需要时加载对应索引
        self.bm25_index_path = bm25_index_path
        self.dense_index_path = dense_index_path
        self.dense_model_name = dense_model_name
        self.corpus_path = corpus_path

        print("[HybridRAGIndex] 初始化完成（懒加载模式）")
        print(f"  - BM25 路径: {bm25_index_path or '未配置'}")
        print(f"  - Dense 路径: {dense_index_path or '未配置'}")

    def _ensure_bm25_loaded(self):
        """确保 BM25 索引已加载"""
        if self.bm25_index is None:
            if not self.bm25_index_path:
                raise RuntimeError("BM25 索引路径未配置，无法执行稀疏检索")
            print("[HybridRAGIndex] 首次使用，正在加载 BM25 索引...")
            self.bm25_index = BM25RAGIndex.load_index(self.bm25_index_path)

    def _ensure_dense_loaded(self):
        """确保 Dense 索引已加载"""
        if self.dense_index is None:
            if not self.dense_index_path or not self.corpus_path:
                raise RuntimeError("Dense 索引路径或语料库路径未配置，无法执行稠密检索")
            print("[HybridRAGIndex] 首次使用，正在加载 Dense 索引...")
            self.dense_index = DenseE5RAGIndex.load_index(
                index_path=self.dense_index_path,
                model_name=self.dense_model_name,
                corpus_path=self.corpus_path,
                device=self.device
            )

    def query(self, query: str, top_k: int = 5, search_type: str = "dense") -> str:
        """
        执行混合检索

        Args:
            query: 查询字符串
            top_k: 返回结果数
            search_type: 检索类型 ("sparse" 或 "dense")
        """
        if search_type == "sparse":
            self._ensure_bm25_loaded()
            return self.bm25_index.query(query, top_k=top_k)
        elif search_type == "dense":
            self._ensure_dense_loaded()
            return self.dense_index.query(query, top_k=top_k)
        else:
            raise ValueError(f"不支持的检索类型: {search_type}，仅支持 'sparse' 或 'dense'")

    # 实现抽象方法（桩方法）
    def build_index(self, *args, **kwargs):
        raise NotImplementedError("HybridRAGIndex 不支持构建索引，请分别构建 BM25 和 Dense 索引")

    def save_index(self, *args, **kwargs):
        pass

    def _store_embeddings(self, *args, **kwargs):
        pass

    def _finalize_embeddings(self, *args, **kwargs):
        pass

    @classmethod
    def load_index(
        cls,
        index_path: str,
        model_name: Optional[str] = None,
        device: str = "cuda",
        bm25_index_path: Optional[str] = None,
        dense_index_path: Optional[str] = None,
        corpus_path: Optional[str] = None,
        **kwargs
    ) -> "HybridRAGIndex":
        """
        加载混合索引

        Args:
            index_path: 基础索引路径（可作为 Dense 索引路径）
            model_name: Dense 模型名称
            device: 运行设备
            bm25_index_path: BM25 索引路径（可选，优先级高于 index_path/bm25）
            dense_index_path: Dense 索引路径（可选，优先级高于 index_path）
            corpus_path: 语料库路径（可选，优先级高于 index_path/corpus.jsonl）
        """
        # 参数优先级处理
        final_bm25_path = bm25_index_path or os.path.join(index_path, "bm25")
        final_dense_path = dense_index_path or index_path
        final_corpus_path = corpus_path or os.path.join(index_path, "corpus.jsonl")
        final_model_name = model_name or "intfloat/e5-base-v2"

        # 检查哪些索引可用
        bm25_available = os.path.exists(final_bm25_path)
        dense_available = os.path.exists(final_dense_path) and os.path.exists(final_corpus_path)

        if not bm25_available and not dense_available:
            raise FileNotFoundError(
                f"未找到可用的索引:\n"
                f"  - BM25: {final_bm25_path}\n"
                f"  - Dense: {final_dense_path} + {final_corpus_path}"
            )

        return cls(
            bm25_index_path=final_bm25_path if bm25_available else None,
            dense_index_path=final_dense_path if dense_available else None,
            dense_model_name=final_model_name,
            corpus_path=final_corpus_path if dense_available else None,
            device=device,
            **kwargs
        )




