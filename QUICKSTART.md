# RAG 环境测评 - 快速开始

## 📋 已创建的文件

1. **run_rag_benchmark.sh** - Bash 启动脚本（推荐）
2. **run_rag_test.py** - Python 启动脚本
3. **demo_rag_test.sh** - 交互式演示脚本
4. **RAG_BENCHMARK_GUIDE.md** - 详细使用指南

## 🚀 快速运行

### 方式 1: 使用默认配置（最简单）

```bash
./run_rag_benchmark.sh
```

这将：
- 使用 `src/data/rag_demo.jsonl` 数据集（5条样本）
- 启动 5 个并行 workers
- 使用 `exact_match` 和 `f1_score` 两种测评指标
- 输出结果到 `results/rag_test_<时间戳>/`

### 方式 2: 使用演示脚本（推荐新手）

```bash
./demo_rag_test.sh
```

这个脚本会：
- 显示数据集样本
- 检查 RAG 配置
- 询问确认后运行测评

### 方式 3: 自定义配置

```bash
# 使用完整 HotPotQA 数据集
DATA_PATH=src/data/HotPotQA.jsonl \
NUM_ROLLOUTS=10 \
OUTPUT_DIR=results/hotpotqa_test \
./run_rag_benchmark.sh
```

### 方式 4: 使用 Python 脚本

```bash
python3 run_rag_test.py
```

## ⚙️ 环境配置

### 必须配置项

1. **启用 RAG 资源**

编辑 [deployment_config.json](deployment_config.json:47-62):
```json
{
  "resources": {
    "rag": {
      "enabled": true,
      ...
    }
  }
}
```

✅ 当前状态: 已启用

2. **设置 API 密钥**

创建 `.env` 文件：
```bash
OPENAI_API_KEY=your_api_key
OPENAI_API_BASE=https://api.openai.com/v1
```

3. **启动资源服务器**

```bash
python3 -m utils.resource_pools.gateway_server
```

## 📊 测评指标

本测评使用两种标准指标：

1. **Exact Match (精确匹配)**
   - 标准化后完全匹配
   - 得分: 0 或 1

2. **F1 Score (F1 分数)**
   - 基于词袋模型的 Precision/Recall
   - 得分: 0.0 到 1.0

## 📁 输出文件

运行完成后，在输出目录中会生成：

```
results/rag_test_<时间戳>/
├── trajectory.jsonl           # 完整执行轨迹
├── evaluation_scores.json     # 详细评分
├── evaluation_summary.json    # 汇总统计
└── worker_instance_map.json   # Worker 映射
```

### 查看结果

```bash
# 查看汇总结果
cat results/rag_test_*/evaluation_summary.json | python3 -m json.tool

# 查看详细评分
cat results/rag_test_*/evaluation_scores.json | python3 -m json.tool
```

## 🔧 可配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| DATA_PATH | src/data/rag_demo.jsonl | 测试数据路径 |
| NUM_ROLLOUTS | 5 | 并行 worker 数 |
| OUTPUT_DIR | results/rag_test_<时间戳> | 输出目录 |
| MODEL_NAME | gpt-4.1-2025-04-14 | LLM 模型 |
| MAX_TURNS | 15 | 最大轮次 |

## 📚 可用数据集

- `src/data/rag_demo.jsonl` - 5条样本（快速测试）
- `src/data/HotPotQA_demo.jsonl` - HotPotQA 样本
- `src/data/HotPotQA.jsonl` - HotPotQA 完整数据集
- `src/data/bamboogle.json` - Bamboogle 数据集

## 💡 使用示例

### 示例 1: 快速测试
```bash
./demo_rag_test.sh
```

### 示例 2: 完整测评
```bash
DATA_PATH=src/data/HotPotQA.jsonl \
NUM_ROLLOUTS=20 \
./run_rag_benchmark.sh
```

### 示例 3: 自定义所有参数
```bash
DATA_PATH=src/data/bamboogle.json \
NUM_ROLLOUTS=10 \
MODEL_NAME=gpt-4.1-2025-04-14 \
MAX_TURNS=20 \
OUTPUT_DIR=results/my_test \
./run_rag_benchmark.sh
```

## ❓ 常见问题

### Q: 如何检查 RAG 是否启用？
```bash
grep -A 2 '"rag"' deployment_config.json | grep enabled
```

### Q: 如何更改 RAG 索引路径？
编辑 `deployment_config.json` 中的 `rag.config.rag_index_path`

### Q: 如何只使用一个测评指标？
修改脚本中的 `--evaluation_metric` 参数

### Q: 如何查看实时日志？
日志会实时输出到控制台

## 📖 详细文档

查看 [RAG_BENCHMARK_GUIDE.md](RAG_BENCHMARK_GUIDE.md) 获取完整文档。

## ⚠️ 注意事项

1. 首次运行需要加载模型和索引（可能较慢）
2. 确保有足够内存加载 RAG 索引
3. 建议先用小数据集测试配置
4. 确保资源服务器已启动

## 🎯 推荐工作流

1. **首次使用**
   ```bash
   # 1. 检查配置
   ./demo_rag_test.sh

   # 2. 如果成功，运行完整测试
   DATA_PATH=src/data/HotPotQA.jsonl ./run_rag_benchmark.sh
   ```

2. **日常使用**
   ```bash
   ./run_rag_benchmark.sh
   ```

3. **大规模测评**
   ```bash
   DATA_PATH=src/data/HotPotQA.jsonl \
   NUM_ROLLOUTS=20 \
   OUTPUT_DIR=results/production_test \
   ./run_rag_benchmark.sh
   ```

---

**提示**: 运行前确保已启动资源服务器并配置环境变量！
