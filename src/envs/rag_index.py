import os
import json
import numpy as np
from typing import Dict, List, Any, Optional
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


# ============================================================================
# RAG 索引基类
# ============================================================================



class BaseRAGIndex(ABC):
    """
    RAG 索引公共基类，负责模型加载、文本解析与embedding生成。
    子类需实现向量存储、索引持久化及查询等特定逻辑。
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cuda", max_seq_length: int = 512) -> None:
        if not _SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("需要安装 sentence_transformers: pip install sentence-transformers")

        print(f"正在加载模型: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        model_max_length = self.model.get_max_seq_length()
        actual_max_length = min(max_seq_length, model_max_length)
        self.model.max_seq_length = actual_max_length
        if actual_max_length < max_seq_length:
            print(f"⚠️  警告: 模型最大支持长度为 {model_max_length}，已将 max_seq_length 从 {max_seq_length} 调整为 {actual_max_length}")
        print(f"✓ 模型已加载到 {device}, 最大序列长度: {actual_max_length}")

        self.chunks: List[Dict[str, Any]] = []
        self.model_name = model_name
        self.device = device
        self.max_seq_length = actual_max_length

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

        if self.device != 'cpu' and num_workers > 1:
            print(f"  ⚠️  GPU模式下不支持多进程，自动切换到单进程")
            num_workers = 1
        else:
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
                 device: str = "cuda", max_seq_length: int = 512):
        super().__init__(model_name=model_name, device=device, max_seq_length=max_seq_length)
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
            "vector_dim": self.vectors.shape[1]
        }
        metadata_file = os.path.join(index_path, "metadata.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 索引已保存到: {index_path}")
        print(f"  - 文本块: {len(self.chunks)}")
        print(f"  - 向量维度: {self.vectors.shape}")

    @classmethod
    def load_index(cls, index_path: str, model_name: Optional[str] = None,
                   device: str = "cuda", max_seq_length: int = 512, **kwargs):
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
            print(f"✓ 从元数据读取配置: {metadata}")
        else:
            if model_name is None:
                model_name = "sentence-transformers/all-MiniLM-L6-v2"
            print("⚠️ 未找到元数据文件,使用默认配置")

        # 创建实例
        instance = cls(model_name=model_name, device=device, max_seq_length=max_seq_length)
        
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
        
        retrieved_chunks = [self.chunks[i] for i in top_k_indices]
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
                 use_gpu_index: bool = False):
        super().__init__(model_name=model_name, device=device, max_seq_length=max_seq_length)
        
        if not _FAISS_AVAILABLE:
            raise ImportError("需要安装 faiss: pip install faiss-cpu 或 faiss-gpu")
        
        self.faiss_index = None
        self.use_gpu_index = use_gpu_index
        self.index_nlist: Optional[int] = None
        self.index_nprobe: Optional[int] = None
        self._faiss_on_gpu = False
        self._gpu_resources = None
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

    def _ensure_faiss_index(self, vector_dim: int, total_vectors: int) -> None:
        if self.faiss_index is None:
            self.faiss_index = self._create_cpu_ivf_index(vector_dim, total_vectors)
            self._faiss_on_gpu = False

    def _train_index_if_needed(self, vectors: np.ndarray) -> None:
        if hasattr(self.faiss_index, "is_trained") and not self.faiss_index.is_trained:
            print("正在训练Faiss索引...")
            self.faiss_index.train(vectors)

    def _ensure_nprobe(self) -> None:
        if self.faiss_index is None or not hasattr(self.faiss_index, "nprobe"):
            return
        if self.index_nprobe is None:
            if hasattr(self.faiss_index, "nlist"):
                self.index_nprobe = min(16, max(1, self.faiss_index.nlist))
            else:
                self.index_nprobe = 1
        self.faiss_index.nprobe = min(self.index_nprobe, getattr(self.faiss_index, "nlist", self.index_nprobe))

    def _finalize_faiss_index(self) -> None:
        if self.faiss_index is None:
            return
        self._ensure_nprobe()
        if self.use_gpu_index and not self._faiss_on_gpu:
            try:
                if self._gpu_resources is None:
                    self._gpu_resources = faiss.StandardGpuResources()
                self.faiss_index = faiss.index_cpu_to_gpu(self._gpu_resources, 0, self.faiss_index)
                self._faiss_on_gpu = True
                if hasattr(self.faiss_index, "nlist"):
                    self.index_nlist = self.faiss_index.nlist
                self._ensure_nprobe()
                print("✓ 索引已迁移到GPU")
            except Exception as e:
                print(f"⚠️ 索引迁移到GPU失败({e}),继续使用CPU索引")
                self._faiss_on_gpu = False
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
            "nprobe": self.index_nprobe
        }
        metadata_file = os.path.join(index_path, "metadata.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Faiss索引已保存到: {index_path}")

    @classmethod
    def load_index(cls, index_path: str, model_name: Optional[str] = None,
                   device: str = "cuda", max_seq_length: int = 512,
                   use_gpu_index: bool = False, **kwargs):
        """
        加载Faiss索引
        
        Args:
            index_path: 索引目录路径
            model_name: 模型名称
            device: 运行设备
            max_seq_length: 最大序列长度
            use_gpu_index: 是否使用GPU索引
        
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
            use_gpu_index=use_gpu_index
        )
        
        # 加载文本块
        with open(chunks_file, 'r', encoding='utf-8') as f:
            instance.chunks = json.load(f)
        
        # 加载Faiss索引
        cpu_index = faiss.read_index(index_file)
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
            
            # 获取结果
            retrieved_chunks = [self.chunks[i] for i in indices if i != -1 and i < len(self.chunks)]
            formatted_chunks = [self._format_chunk_for_output(chunk) for chunk in retrieved_chunks]
            context = "\n---\n".join(formatted_chunks)
            
            return f"### Retrieved Context:\n{context}"
        except Exception as e:
            return f"[查询错误] {str(e)}"


def get_rag_index_class(use_faiss: bool = False):
    """根据配置获取本地RAG索引类"""
    if not _SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError("sentence_transformers 未安装, 无法使用本地RAG索引")

    if use_faiss:
        if not _FAISS_AVAILABLE:
            print("⚠️ Faiss 不可用,回退到 RAGIndexLocal (Numpy 实现)")
            return RAGIndexLocal
        print("✅ 使用 RAGIndexLocal_faiss (本地embedding + Faiss加速)")
        return RAGIndexLocal_faiss

    print("✅ 使用 RAGIndexLocal (本地embedding)")
    return RAGIndexLocal
