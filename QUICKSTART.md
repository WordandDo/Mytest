# Quick Start Guide

## 快速开始

### 1. 运行单个基准测试

最简单的方式是直接运行对应的脚本：

```bash
# Dense-only RAG (使用 E5 语义检索)
./benchmark_dense.sh

# Sparse-only RAG (使用 BM25 关键词检索)
./benchmark_sparse.sh

# Hybrid RAG (同时使用两种检索方法)
./benchmark_hybrid.sh

# No Tool (纯 LLM，无检索工具)
./benchmark_no_tool.sh
```

### 2. 运行所有基准测试

```bash
./run_all_benchmarks.sh
```

这将按顺序运行所有四个基准测试。

### 3. 查看结果

测试完成后，结果保存在 `results/` 目录下：

```bash
ls results/
# benchmark_dense_only/
# benchmark_sparse_only/
# benchmark_hybrid/
# benchmark_no_tool/
```

---

## 配置说明

### 默认配置

每个脚本的默认配置：

| 参数 | 默认值 |
|-----|-------|
| 数据集 | `src/data/bamboogle.json` |
| Rollouts | 10 (dense/sparse/no_tool), 5 (hybrid) |
| 模型 | `openai/gpt-oss-120b` |
| 最大轮次 | 15 |
| 评估指标 | exact_match, f1_score |

### 修改配置

在运行脚本前设置环境变量：

```bash
# 修改数据集
export DATA_PATH="src/data/my_dataset.json"

# 修改 rollouts 数量
export NUM_ROLLOUTS=20

# 修改模型
export MODEL_NAME="gpt-4"

# 运行测试
./benchmark_hybrid.sh
```

---

## 系统要求

### 必需组件

1. **Python 3.7+**
2. **MCP Server** (由脚本自动启动)
3. **Resource API** (如果需要资源分配)

### 端口要求

- **8080**: MCP Gateway (自动清理旧进程)
- **8000**: Resource API (如果配置了 `RESOURCE_API_URL`)

### API Keys

确保已配置环境变量或 `.env` 文件：
```bash
OPENAI_API_KEY=your_key_here
# 或其他 LLM 提供商的 API key
```

---

## 故障排查

### 问题：端口 8080 被占用

```bash
# 手动清理端口
lsof -ti:8080 | xargs kill -9
```

### 问题：Gateway 启动失败

检查配置文件是否存在：
```bash
ls gateway_config_rag_*.json
```

查看 Python 错误信息：
```bash
python src/mcp_server/main.py --config gateway_config_rag_hybrid.json --port 8080
```

### 问题：找不到数据文件

检查数据文件路径：
```bash
ls src/data/bamboogle.json
```

或使用自己的数据文件：
```bash
export DATA_PATH="path/to/your/data.json"
./benchmark_dense.sh
```

---

## 下一步

1. ✅ 运行基准测试
2. 📊 分析结果（查看 `results/` 目录）
3. 📖 阅读详细文档：
   - [BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md) - 完整使用指南
   - [BENCHMARK_COMPARISON.md](BENCHMARK_COMPARISON.md) - 配置对比和实验设计
   - [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - 实现细节

---

## 获取帮助

- 查看详细文档：`BENCHMARK_GUIDE.md`
- 查看实现细节：`IMPLEMENTATION_SUMMARY.md`
- 查看配置对比：`BENCHMARK_COMPARISON.md`
