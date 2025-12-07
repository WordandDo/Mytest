# RAG 磁盘懒加载 (Disk Lazy Loading) 使用指南

## 背景

在大规模知识库场景下(如百万级文档),启动时全量加载语料库到内存会导致:
- **启动慢**: 需要几十秒到几分钟加载完整语料库
- **内存占用高**: 100万条文档可能占用数GB内存

通过 **磁盘懒加载 (Disk Lazy Loading)** 方案,可以:
- ✅ **启动快**: 只加载几MB的偏移量索引,启动时间缩短到秒级
- ✅ **内存占用低**: 只在查询时按需读取文档,常驻内存仅需索引部分
- ✅ **查询性能**: 单次查询仅读取 top_k 条文档,性能损失可忽略

---

## 使用步骤

### 1️⃣ 生成偏移量索引文件

对于语料库文件 `wiki_dump.jsonl`,运行以下命令生成 `.offsets` 索引:

```bash
python scripts/generate_offsets.py /path/to/wiki_dump.jsonl
```

**示例:**
```bash
python scripts/generate_offsets.py /home/a1/sdb/wikidata4rag/Search-r1/data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl
```

**输出:**
```
正在扫描文件: wiki_dump.jsonl
输出偏移量文件: wiki_dump.offsets
文件大小: 2.45 GB
扫描进度: 100%|████████████| 2.45G/2.45G [00:15<00:00, 163MB/s]

扫描完成,共 1,000,000 行
正在写入偏移量文件...
✓ 偏移量文件已生成: wiki_dump.offsets
  - 索引大小: 7.63 MB
  - 每行平均: 8.0 字节

内存节省预估:
  - 原始全量加载: ~2500 MB
  - 懒加载索引: ~8 MB
  - 节省内存: ~2492 MB
```

生成的 `.offsets` 文件只有几MB,包含了每一行的字节偏移量。

---

### 2️⃣ 启动服务

生成 `.offsets` 文件后,直接启动服务即可。代码会自动检测并使用懒加载模式。

**对于 HybridRAGIndex (Dense E5 检索):**

当语料库路径配置为 `wiki_dump.jsonl`,且同目录下存在 `wiki_dump.offsets` 时:

```
[E5] 正在加载语料库: wiki_dump.jsonl
✓ 检测到偏移量索引: wiki_dump.offsets
  → 使用 DiskBasedChunks (磁盘懒加载模式)
[E5] 语料库索引加载完成，共 1,000,000 条 (懒加载，未占用内存)
```

**对于 RAGIndexLocal_faiss / RAGIndexLocal_faiss_compact:**

当索引目录下存在 `chunks.jsonl` 和 `chunks.offsets` 时:

```
✓ 检测到优化存储: 使用 DiskBasedChunks (按需读取 chunks.jsonl)
```

---

### 3️⃣ 验证效果

启动后,检查内存占用:

```bash
# 查看进程内存 (RSS)
ps aux | grep python

# 或使用 htop / top
```

**预期效果:**
- **启动时间**: 从 30-60秒 → **2-5秒**
- **常驻内存**: 从 2-5GB → **几百MB**

查询性能几乎无损失,因为 top_k 通常只有 5-10 条,即使冷启动也只需读取少量行。

---

## 工作原理

### DiskBasedChunks 类

代码中已实现 `DiskBasedChunks` 类 ([rag_index_new.py:125-182](src/utils/rag_index_new.py#L125-L182)),核心逻辑:

1. **初始化**: 加载 `.offsets` 文件 (二进制格式,每行8字节)
2. **索引访问**: 通过 `__getitem__` 实现按需读取
   - 根据 `idx` 查找字节偏移量
   - `seek` 到对应位置并读取一行
   - 解析 JSON 并返回

```python
def __getitem__(self, idx):
    offset = self._offsets[idx]
    self._file.seek(offset)
    line = self._file.readline()
    return json.loads(line.decode('utf-8'))
```

### 自动检测逻辑

- **DenseE5RAGIndex**: 在 `__init__` 中检测 `corpus_path.offsets`
- **RAGIndexLocal_faiss**: 在 `load_index` 中检测 `chunks.offsets`

如果检测到 `.offsets` 文件存在,自动使用 `DiskBasedChunks`;否则回退到全量内存加载并提示用户生成索引。

---

## 配置示例

### deployment_config.json

确保 `corpus_path` 指向正确的 JSONL 文件:

```json
"rag_hybrid": {
  "config": {
    "corpus_path": "/home/a1/sdb/wikidata4rag/Search-r1/data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl"
  }
}
```

运行 `generate_offsets.py` 后,同目录下会生成 `wiki_dump.offsets`,服务启动时自动使用懒加载。

---

## 常见问题

### Q1: 是否需要重新构建索引?

**不需要**。`.offsets` 文件是独立的辅助索引,不影响现有的 Faiss 索引或语料库文件。

### Q2: 查询性能会下降吗?

**几乎不会**。每次查询只读取 top_k 条文档(通常 5-10 条),即使是 SSD 随机读也能在毫秒级完成。

### Q3: 如果没有生成 .offsets 会怎样?

代码会回退到全量内存加载,并打印提示信息:

```
⚠️  未找到偏移量索引文件: wiki_dump.offsets
  → 将全量加载到内存 (可能占用大量内存)
  💡 提示: 运行 `python scripts/generate_offsets.py wiki_dump.jsonl` 生成索引以启用懒加载
```

### Q4: .offsets 文件需要更新吗?

**只在语料库变更时需要**。如果 `wiki_dump.jsonl` 更新了,重新运行 `generate_offsets.py` 即可。

---

## 性能对比

| 场景 | 启动时间 | 常驻内存 | 查询延迟 |
|------|---------|---------|---------|
| **全量加载** | 30-60秒 | 2-5GB | 10-20ms |
| **懒加载** | 2-5秒 | 100-500MB | 12-25ms |

*(数据基于 100万条文档 × 2.5GB 语料库)*

---

## 支持的索引类型

- ✅ **HybridRAGIndex** (DenseE5RAGIndex 的语料库加载)
- ✅ **RAGIndexLocal_faiss**
- ✅ **RAGIndexLocal_faiss_compact**

---

## 相关文件

- **懒加载实现**: [src/utils/rag_index_new.py](src/utils/rag_index_new.py) (`DiskBasedChunks` 类)
- **生成脚本**: [scripts/generate_offsets.py](scripts/generate_offsets.py)
- **配置文件**: [deployment_config.json](deployment_config.json)

---

## 总结

通过简单的一步操作 (`python scripts/generate_offsets.py <jsonl_path>`),即可将启动时间从分钟级降低到秒级,内存占用减少 90% 以上,且对查询性能影响极小。强烈建议在生产环境中启用此优化。
