# RAG 多模式测评指南

## 概述

本指南说明如何使用 `run_rag_env_multimode.sh` 脚本运行三种不同的 RAG 检索模式。

## 三种检索模式

### 1. 混合模式 (hybrid)
- **工具**: 同时提供稀疏检索 (BM25) 和密集检索 (E5)
- **配置文件**: `gateway_config_rag_hybrid.json`
- **适用场景**: Agent 可以根据查询类型自主选择最优检索方法
- **优势**: 灵活性最高，可以处理各种类型的查询

### 2. 仅密集检索模式 (dense)
- **工具**: 仅提供密集向量检索 (E5/Contriever)
- **配置文件**: `gateway_config_rag_dense_only.json`
- **适用场景**: 语义理解、概念搜索、自然语言问题
- **优势**: 擅长理解查询意图，查找语义相似内容

### 3. 仅稀疏检索模式 (sparse)
- **工具**: 仅提供关键词匹配 (BM25)
- **配置文件**: `gateway_config_rag_sparse_only.json`
- **适用场景**: 精确词匹配、ID/代码搜索、特定短语查找
- **优势**: 对具体术语和实体名称的匹配精准度高

## 使用方法

### 基本用法

```bash
# 运行所有三种模式（默认）
./run_rag_env_multimode.sh all

# 仅运行混合模式
./run_rag_env_multimode.sh hybrid

# 仅运行密集检索模式
./run_rag_env_multimode.sh dense

# 仅运行稀疏检索模式
./run_rag_env_multimode.sh sparse

# 查看帮助
./run_rag_env_multimode.sh --help
```

### 自定义参数

通过环境变量配置运行参数：

```bash
# 使用自定义数据集和并行数
DATA_PATH="src/data/custom.jsonl" NUM_ROLLOUTS=20 ./run_rag_env_multimode.sh all

# 使用不同模型
MODEL_NAME="gpt-4" ./run_rag_env_multimode.sh hybrid

# 自定义评测指标
EVALUATION_METRICS="exact_match f1_score" ./run_rag_env_multimode.sh dense

# 完整自定义示例
DATA_PATH="src/data/mydata.jsonl" \
NUM_ROLLOUTS=15 \
MODEL_NAME="gpt-4-turbo" \
MAX_TURNS=20 \
EVALUATION_METRICS="exact_match f1_score similarity" \
BASE_OUTPUT_DIR="results/my_experiment" \
./run_rag_env_multimode.sh all
```

## 支持的环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `DATA_PATH` | `src/data/HotPotQA.jsonl` | 测评数据集路径 |
| `NUM_ROLLOUTS` | `10` | 并行任务数量 |
| `MODEL_NAME` | `gpt-4.1-2025-04-14` | 使用的语言模型 |
| `MAX_TURNS` | `15` | 每个任务最大对话轮次 |
| `MCP_SERVER_URL` | `http://localhost:8080` | MCP 服务器地址 |
| `TASK_EXECUTION_TIMEOUT` | `900` | 任务超时时间（秒） |
| `EVALUATION_METRICS` | `exact_match f1_score similarity contains_answer` | 评测指标列表 |
| `BASE_OUTPUT_DIR` | `results/rag_multimode` | 输出基础目录 |

## 输出结构

运行脚本后，结果将保存在以下目录结构中：

```
results/rag_multimode/
├── hybrid_20251207_143025/          # 混合模式结果
│   ├── config.txt                   # 运行配置
│   ├── evaluation_scores.json       # 详细评分
│   ├── evaluation_summary.json      # 汇总统计
│   └── trajectory.jsonl             # 执行轨迹
├── dense_20251207_143025/           # 密集检索模式结果
│   ├── config.txt
│   ├── evaluation_scores.json
│   ├── evaluation_summary.json
│   └── trajectory.jsonl
├── sparse_20251207_143025/          # 稀疏检索模式结果
│   ├── config.txt
│   ├── evaluation_scores.json
│   ├── evaluation_summary.json
│   └── trajectory.jsonl
└── summary_20251207_143025.txt      # 汇总报告
```

### 关键输出文件

- **config.txt**: 记录本次运行的所有配置参数
- **evaluation_scores.json**: 每个测试任务的详细评分结果
- **evaluation_summary.json**: 所有指标的统计汇总（平均分、成功率等）
- **trajectory.jsonl**: 完整的 Agent 执行轨迹（包含工具调用、推理过程）
- **summary_TIMESTAMP.txt**: 多模式运行的总体汇总报告

## 工作原理

脚本通过切换不同的 Gateway 配置文件来控制 Agent 可用的检索工具：

1. **gateway_config_rag_hybrid.json**
   - 暴露 `rag_query` (密集检索) 和 `rag_query_sparse` (稀疏检索)
   - Agent 可以自主选择使用哪种检索方法

2. **gateway_config_rag_dense_only.json**
   - 仅暴露 `rag_query` (密集检索)
   - Agent 只能使用向量语义检索

3. **gateway_config_rag_sparse_only.json**
   - 仅暴露 `rag_query_sparse` (稀疏检索)
   - Agent 只能使用 BM25 关键词检索

## 示例场景

### 场景 1: 对比三种模式性能

```bash
# 在相同数据集上运行所有三种模式
DATA_PATH="src/data/HotPotQA.jsonl" \
NUM_ROLLOUTS=20 \
./run_rag_env_multimode.sh all

# 结果将分别保存，便于对比分析
```

### 场景 2: 快速测试新数据集

```bash
# 使用小规模快速测试
DATA_PATH="src/data/test_sample.jsonl" \
NUM_ROLLOUTS=5 \
MAX_TURNS=10 \
./run_rag_env_multimode.sh hybrid
```

### 场景 3: 生产环境正式评测

```bash
# 大规模正式评测
DATA_PATH="src/data/HotPotQA.jsonl" \
NUM_ROLLOUTS=50 \
MODEL_NAME="gpt-4.1-2025-04-14" \
MAX_TURNS=20 \
TASK_EXECUTION_TIMEOUT=1800 \
BASE_OUTPUT_DIR="results/production_eval" \
./run_rag_env_multimode.sh all
```

## 前置条件

1. **MCP 服务器必须运行**
   ```bash
   # 确保 MCP 服务器在 localhost:8080 运行
   # 或通过 MCP_SERVER_URL 指定其他地址
   ```

2. **RAG 索引已配置**
   - 在 `deployment_config.json` 中配置了 RAG 资源池
   - 索引文件路径正确且可访问

3. **测评数据集存在**
   ```bash
   # 确认数据文件存在
   ls -lh src/data/HotPotQA.jsonl
   ```

## 常见问题

### Q: 如何只运行一个模式？
A: 直接指定模式名称，例如 `./run_rag_env_multimode.sh dense`

### Q: 如何自定义输出目录？
A: 使用 `BASE_OUTPUT_DIR` 环境变量，例如 `BASE_OUTPUT_DIR="my_results" ./run_rag_env_multimode.sh all`

### Q: 可以中途停止吗？
A: 可以，按 Ctrl+C 停止。已完成的模式结果会保留。

### Q: 如何对比不同模式的结果？
A: 查看各模式目录下的 `evaluation_summary.json` 文件，对比 average_score 和 success_rate 等指标。

## 相关文件

- 主脚本: [run_rag_env_multimode.sh](run_rag_env_multimode.sh)
- Gateway 配置:
  - [gateway_config_rag_hybrid.json](gateway_config_rag_hybrid.json)
  - [gateway_config_rag_dense_only.json](gateway_config_rag_dense_only.json)
  - [gateway_config_rag_sparse_only.json](gateway_config_rag_sparse_only.json)
- 原始脚本: [run_rag_env.sh](run_rag_env.sh)
- Python 实现: [run_rag_test.py](run_rag_test.py)
