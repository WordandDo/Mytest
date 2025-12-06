# 混合检索 (Hybrid RAG) 使用指南

## 概述

本系统现已支持**混合检索模式**，可以在同一个 RAG 服务中同时使用：
- **Sparse 检索 (BM25)**: 基于关键词的精确匹配
- **Dense 检索 (E5)**: 基于语义向量的相似度搜索

## 架构变更说明

### 移除的功能
- ❌ **GainRAG 索引**: 已被新的混合检索方案完全替代
  - 原因：与新方案功能重叠，且配置复杂度高
  - 迁移：可使用 Dense 检索 + E5 模型达到类似效果

### 新增功能
- ✅ **HybridRAGIndex**: 统一的混合检索索引类
- ✅ **懒加载机制**: 仅在首次使用时加载对应索引，减少内存占用
- ✅ **双工具接口**: Agent 可以根据任务选择最合适的检索方式

## 使用方法

### 1. 准备索引文件

#### 1.1 BM25 索引（使用 Pyserini 构建）

```bash
# 安装 Pyserini（需要 Java 环境）
pip install pyserini

# 构建 BM25 索引
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input /path/to/corpus \
  --index /path/to/bm25_index \
  --generator DefaultLuceneDocumentGenerator \
  --threads 8 \
  --storePositions --storeDocvectors --storeRaw
```

语料格式示例（JSONL）:
```json
{"id": "doc1", "contents": "这是文档内容", "title": "文档标题"}
{"id": "doc2", "contents": "另一个文档", "title": "标题2"}
```

#### 1.2 Dense 索引（使用 Faiss + E5）

```python
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel

# 1. 加载 E5 模型
model_name = "intfloat/e5-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 2. 编码语料库（示例）
def encode_corpus(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.detach().numpy()

# 3. 构建 Faiss 索引
embeddings = encode_corpus(corpus_texts)  # corpus_texts: List[str]
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner Product (余弦相似度)
faiss.normalize_L2(embeddings)
index.add(embeddings)

# 4. 保存索引
faiss.write_index(index, "/path/to/dense_index/e5_Flat.index")
```

### 2. 配置 deployment_config.json

```json
{
  "resources": {
    "rag_hybrid": {
      "enabled": true,
      "implementation_class": "utils.resource_pools.rag_pool.RAGPoolImpl",
      "config": {
        "num_rag_workers": 50,

        "use_hybrid": true,

        "bm25_index_path": "/path/to/bm25_index",
        "dense_index_path": "/path/to/dense_index/e5_Flat.index",
        "corpus_path": "/path/to/corpus.jsonl",
        "rag_model_name": "intfloat/e5-base-v2",

        "embedding_device": "cuda",
        "use_gpu_index": true,
        "default_top_k": 10,

        "server_start_retries": 600
      }
    }
  }
}
```

**配置说明**:
- `use_hybrid`: **必须设为 true** 启用混合检索
- `bm25_index_path`: BM25 索引目录（可选，不配置则 Sparse 检索不可用）
- `dense_index_path`: Dense 索引文件路径（可选，不配置则 Dense 检索不可用）
- `corpus_path`: 原始语料库 JSONL 文件（Dense 检索必需）
- `rag_model_name`: E5 模型名称（默认 `intfloat/e5-base-v2`）

### 3. 在 Agent 中使用

#### 方式 1: 使用默认工具（Dense 检索）

```python
from mcp_server.rag_server import query_knowledge_base_dense

# 默认使用语义检索
result = await query_knowledge_base_dense(
    worker_id="worker_1",
    query="什么是量子计算?",
    top_k=5
)
```

#### 方式 2: 使用 Sparse 检索

```python
from mcp_server.rag_server import query_knowledge_base_sparse

# 精确关键词匹配
result = await query_knowledge_base_sparse(
    worker_id="worker_1",
    query="量子比特 qubit",
    top_k=5
)
```

#### 方式 3: 在 MCP 工具描述中选择

Agent 会根据任务自动选择合适的工具：

```markdown
可用工具:
1. **query_knowledge_base_dense** [rag_query]
   - 语义检索，适合概念搜索和自然语言问题

2. **query_knowledge_base_sparse** [rag_query_sparse]
   - 关键词检索，适合精确术语匹配和 ID 查找
```

## 工具选择建议

### 使用 Dense 检索（语义搜索）的场景
- ✅ 自然语言问题："量子计算的原理是什么？"
- ✅ 概念搜索："与机器学习相关的内容"
- ✅ 模糊查询："提高性能的方法"
- ✅ 跨语言检索（如果模型支持）

### 使用 Sparse 检索（关键词匹配）的场景
- ✅ 精确术语匹配："HTTP 404 错误"
- ✅ 代码/ID 查找："find user_id: 12345"
- ✅ 特定名称："Apple Inc. 财报"
- ✅ 缩写词："COVID-19 疫苗"

## 懒加载机制

系统采用**懒加载**策略，优化内存使用：

```python
# 启动时只初始化路径信息
hybrid_index = HybridRAGIndex(
    bm25_index_path="/path/to/bm25",
    dense_index_path="/path/to/dense",
    corpus_path="/path/to/corpus.jsonl"
)
# 此时尚未加载任何索引

# 首次使用 Dense 检索时自动加载
result = hybrid_index.query("query text", search_type="dense")
# 触发加载: DecExEncoder + Faiss Index + Corpus

# 首次使用 Sparse 检索时自动加载
result = hybrid_index.query("query text", search_type="sparse")
# 触发加载: Pyserini BM25 Searcher
```

**优势**:
- 减少启动时间
- 降低内存占用（如果只使用一种检索方式）
- 支持仅配置单一索引的场景

## 依赖安装

```bash
# 基础依赖
pip install torch transformers faiss-gpu numpy tqdm

# Sparse 检索依赖（需要 Java 环境）
pip install pyserini

# 验证 Java 环境
java -version  # 需要 Java 11+
```

## 故障排查

### 问题 1: `BM25 索引路径未配置`

**原因**: 调用了 `query_knowledge_base_sparse` 但未配置 `bm25_index_path`

**解决**:
```json
{
  "config": {
    "use_hybrid": true,
    "bm25_index_path": "/valid/path/to/bm25"
  }
}
```

### 问题 2: `ImportError: pyserini 未安装`

**原因**: BM25 检索需要 Pyserini 库

**解决**:
```bash
# 确保安装了 Java
sudo apt-get install openjdk-11-jdk

# 安装 Pyserini
pip install pyserini
```

### 问题 3: `Dense 索引路径或语料库路径未配置`

**原因**: 调用了 `query_knowledge_base_dense` 但缺少必需配置

**解决**:
```json
{
  "config": {
    "use_hybrid": true,
    "dense_index_path": "/path/to/faiss.index",
    "corpus_path": "/path/to/corpus.jsonl"
  }
}
```

### 问题 4: GPU 内存不足

**解决**: 使用 CPU 或调整 batch size
```json
{
  "config": {
    "embedding_device": "cpu",
    "use_gpu_index": false
  }
}
```

## 性能优化建议

1. **索引预热**: 启动后主动调用一次查询，触发索引加载
2. **GPU 加速**:
   - E5 模型推理使用 GPU
   - Faiss 索引迁移到 GPU (设置 `use_gpu_index: true`)
3. **语料压缩**: 如果语料库过大，考虑使用 mmap 或数据库存储
4. **混合检索**: 结合 Dense 和 Sparse 结果进行重排序（需自行实现）

## 与旧版 GainRAG 的区别

| 特性 | GainRAG | Hybrid RAG |
|------|---------|------------|
| 模型支持 | Contriever 固定 | E5/任意 Transformer |
| 检索方式 | 仅 Dense | Dense + Sparse |
| 内存占用 | 高（始终加载） | 低（懒加载） |
| 配置复杂度 | 高 | 中 |
| 扩展性 | 低 | 高 |

## 下一步

- [ ] 实现混合检索结果的重排序（Reciprocal Rank Fusion）
- [ ] 支持更多稠密检索模型（BGE, Instructor 等）
- [ ] 添加检索结果缓存机制
- [ ] 支持多语言检索

## 贡献

如有问题或建议，请提交 Issue 或 Pull Request。
