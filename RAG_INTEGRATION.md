# RAG Index 集成说明

## 概述

已成功将新的 `rag_index_new.py` 集成到现有的 `rag_pool.py` 中。新版本支持以下高级特性：

- ✅ **GainRAG 索引**: 使用 Contriever 模型的高级 RAG 实现
- ✅ **压缩索引 (IVFPQ)**: 大幅降低索引体积，适合大规模知识库
- ✅ **多GPU支持**: 支持多卡编码和跨卡索引并行
- ✅ **向后兼容**: 完全兼容原有配置

## 主要修改

### 1. 更新导入语句

```python
# 旧版本
from utils.rag_index import get_rag_index_class, BaseRAGIndex

# 新版本
from utils.rag_index_new import get_rag_index_class, BaseRAGIndex
```

### 2. 增强的 `start_rag_server` 函数

新版本的函数支持以下适配点：

#### 适配点 1: 基础路径配置
- `rag_kb_path`: 知识库文件路径
- `rag_index_path`: 索引保存/加载路径
- `rag_model_name`: Embedding 模型名称
- `embedding_device`: 主要的 embedding 设备

#### 适配点 2: 类型开关（布尔值解析）
- `use_faiss`: 是否使用 Faiss 加速
- `use_gpu_index`: 是否将 Faiss 索引放在 GPU 上
- `use_compact`: 是否使用压缩索引 (IVFPQ)
- `use_gainrag`: 是否使用 GainRAG 模式

支持字符串和布尔值格式（如 "true", "1", "yes" 都会被识别为 True）

#### 适配点 3: 高级参数
- `gpu_parallel_degree`: Faiss 索引跨多少张 GPU 卡
- `embedding_devices`: 编码时使用的设备列表（支持逗号分隔字符串或列表）
- `target_bytes_per_vector`: 压缩索引的目标字节数/向量
- `passages_path`: Passages 文件路径（GainRAG 必需）
- `gpu_id`: GPU 设备 ID（用于 GainRAG）

#### 适配点 4: 工厂函数调用
```python
IndexClass = get_rag_index_class(
    use_faiss=use_faiss,
    use_compact=use_compact,
    use_gainrag=use_gainrag
)
```

#### 适配点 5: 参数字典构建
根据不同的索引类型，自动注入相应的参数：
- Faiss 体系: `use_gpu_index`, `gpu_parallel_degree`
- Compact 索引: `target_bytes_per_vector`, `memory_map`
- GainRAG: `gpu_id`, `passages_path`

#### 适配点 6: 加载/构建逻辑
- 自动检测索引类型（metadata.json 或 index.faiss）
- GainRAG 模式不支持在线构建，必须提供预构建的索引
- 标准模式支持在线构建，使用多进程加速

## 配置示例

### 示例 1: 标准 Faiss 索引

```json
{
  "rag": {
    "enabled": true,
    "implementation_class": "utils.resource_pools.rag_pool.RAGPoolImpl",
    "config": {
      "num_rag_workers": 3,
      "rag_kb_path": "/data/knowledge.jsonl",
      "rag_index_path": "/data/index_standard",
      "rag_model_name": "sentence-transformers/all-MiniLM-L6-v2",
      "embedding_device": "cuda:0",

      "use_faiss": true,
      "use_gpu_index": true,
      "default_top_k": 10
    }
  }
}
```

### 示例 2: 高性能压缩索引 + 多GPU编码

```json
{
  "rag": {
    "enabled": true,
    "implementation_class": "utils.resource_pools.rag_pool.RAGPoolImpl",
    "config": {
      "num_rag_workers": 3,
      "rag_kb_path": "/data/knowledge.jsonl",
      "rag_index_path": "/data/index_compact",
      "rag_model_name": "sentence-transformers/all-MiniLM-L6-v2",
      "embedding_device": "cuda:0",

      "use_faiss": true,
      "use_compact": true,
      "use_gpu_index": true,
      "gpu_parallel_degree": 2,
      "embedding_devices": "cuda:0,cuda:1",
      "target_bytes_per_vector": 32,
      "default_top_k": 10
    }
  }
}
```

### 示例 3: GainRAG 模式

```json
{
  "rag": {
    "enabled": true,
    "implementation_class": "utils.resource_pools.rag_pool.RAGPoolImpl",
    "config": {
      "num_rag_workers": 3,
      "rag_index_path": "/data/gainrag_index_folder",
      "rag_model_name": "facebook/contriever",
      "passages_path": "/data/passages.jsonl",

      "use_gainrag": true,
      "use_gpu_index": true,
      "gpu_id": 0,
      "default_top_k": 10
    }
  }
}
```

### 示例 4: 基础本地 Numpy 索引

```json
{
  "rag": {
    "enabled": true,
    "implementation_class": "utils.resource_pools.rag_pool.RAGPoolImpl",
    "config": {
      "num_rag_workers": 2,
      "rag_kb_path": "/data/small_knowledge.jsonl",
      "rag_index_path": "/data/index_numpy",
      "rag_model_name": "sentence-transformers/all-MiniLM-L6-v2",
      "embedding_device": "cpu",

      "use_faiss": false,
      "default_top_k": 5
    }
  }
}
```

## 参数优先级

### 布尔值解析
支持以下格式：
- Python 布尔值: `true`, `false`
- 字符串: `"true"`, `"false"`, `"1"`, `"0"`, `"yes"`, `"no"`

### embedding_devices 解析
支持以下格式：
- 逗号分隔字符串: `"cuda:0,cuda:1"`
- Python 列表: `["cuda:0", "cuda:1"]`

### 参数传递优先级
1. **模型主设备**: `embedding_device` → `common_kwargs['device']`
2. **多卡编码**: `embedding_devices` → `common_kwargs['embedding_devices']`
3. **GPU索引**: `use_gpu_index` → Faiss/GainRAG 专用参数
4. **并行度**: `gpu_parallel_degree` → Faiss 专用参数

## 特殊情况处理

### GainRAG 模式
- ❌ **不支持在线构建**: 必须提供预构建的索引路径
- ✅ **必需文件**: `index.faiss` 和 `passages.jsonl`
- ✅ **自动检测**: 在 index_path 下自动查找 passages 文件

### 压缩索引 (Compact)
- ✅ **自动内存映射**: 设置 `memory_map=True` 以减少内存占用
- ✅ **自动参数推导**: 如果未指定 `pq_m`，会根据 `target_bytes_per_vector` 自动计算

### 多GPU配置
- ⚠️ **编码多GPU**: `embedding_devices` 用于多卡并行编码（需要 sentence-transformers 支持）
- ⚠️ **索引多GPU**: `gpu_parallel_degree` 用于 Faiss 索引跨卡并行查询
- ⚠️ **互斥性**: 多GPU编码时会忽略 `num_workers` 参数

## 向后兼容性

新版本完全兼容旧配置：

```json
{
  "rag": {
    "config": {
      "rag_index_path": "/data/old_index",
      "rag_model_name": "sentence-transformers/all-MiniLM-L6-v2",
      "use_faiss": true,
      "embedding_device": "cpu"
    }
  }
}
```

旧配置会自动使用默认值：
- `use_compact = false`
- `use_gainrag = false`
- `embedding_devices = None`
- `gpu_parallel_degree = None`

## 故障排查

### 问题 1: 导入错误
```
ModuleNotFoundError: No module named 'utils.rag_index_new'
```

**解决方案**: 确保 `rag_index_new.py` 在 `src/utils/` 目录下

### 问题 2: GainRAG 构建失败
```
RuntimeError: GainRAGIndex does not support online building
```

**解决方案**: GainRAG 模式只能加载预构建的索引，需要提供有效的 `index_path`

### 问题 3: embedding_devices 解析失败
```
TypeError: 'NoneType' object is not iterable
```

**解决方案**: 检查配置文件中 `embedding_devices` 的格式，应为逗号分隔字符串或列表

### 问题 4: GPU 索引失败
```
RuntimeError: CUDA error: out of memory
```

**解决方案**:
1. 减少 `gpu_parallel_degree`
2. 启用 `use_compact=true` 使用压缩索引
3. 设置 `use_gpu_index=false` 使用 CPU 索引

## 性能优化建议

### 小规模数据 (< 10万条)
- 使用 `RAGIndexLocal` (不启用 Faiss)
- 设置 `use_faiss=false`

### 中等规模数据 (10万 - 100万条)
- 使用 `RAGIndexLocal_faiss`
- 设置 `use_faiss=true`
- 可选启用 `use_gpu_index=true`

### 大规模数据 (> 100万条)
- 使用 `RAGIndexLocal_faiss_compact`
- 设置 `use_faiss=true`, `use_compact=true`
- 建议启用 `use_gpu_index=true`
- 设置合适的 `target_bytes_per_vector` (24-64)

### 多GPU环境
- 启用 `embedding_devices` 加速编码
- 启用 `gpu_parallel_degree` 加速查询
- 建议 `gpu_parallel_degree` ≤ 可用GPU数量

## 文件清单

### 修改的文件
- ✅ `src/utils/resource_pools/rag_pool.py`
  - 更新导入语句
  - 重写 `start_rag_server` 函数
  - 增加参数解析逻辑

### 新增的文件
- ✅ `deployment_config_examples.json` - 配置示例文件
- ✅ `RAG_INTEGRATION.md` - 本说明文档

### 依赖的文件
- `src/utils/rag_index_new.py` - 新的索引实现（应已存在）
- `deployment_config.json` - 部署配置文件

## 测试建议

### 基础测试
```bash
# 1. 检查语法
python3 -m py_compile src/utils/resource_pools/rag_pool.py

# 2. 启动服务（使用旧配置）
# 应该能正常启动，保持向后兼容

# 3. 测试新特性（更新配置后）
# 逐步启用 use_compact, embedding_devices 等新参数
```

### 功能测试
1. **标准模式**: 使用旧配置验证向后兼容性
2. **Compact模式**: 启用压缩索引验证内存占用
3. **多GPU模式**: 测试多卡编码和查询性能
4. **GainRAG模式**: 加载预构建的 GainRAG 索引

## 总结

集成工作已完成，主要改进：

1. ✅ **功能扩展**: 支持 GainRAG、压缩索引、多GPU
2. ✅ **参数灵活**: 智能解析布尔值、列表等多种格式
3. ✅ **向后兼容**: 完全兼容旧配置
4. ✅ **错误处理**: 增加了更多的异常捕获和日志
5. ✅ **文档完善**: 提供详细的配置示例和说明

新版本可以直接替换旧版本使用，无需修改现有配置。如需使用新特性，参考本文档中的配置示例即可。
