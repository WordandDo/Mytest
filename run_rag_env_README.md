# run_rag_env.sh 使用指南

## 概述

`run_rag_env.sh` 脚本用于运行 RAG-only 环境测评，现已支持多个评测指标。

## 快速开始

### 基本用法（使用默认配置）

```bash
./run_rag_env.sh
```

默认配置：
- **评测指标**: exact_match, f1_score, similarity, contains_answer（4个指标）
- **数据集**: src/data/HotPotQA_demo.jsonl
- **并行度**: 50
- **模型**: gpt-4.1-2025-04-14
- **最大轮次**: 15
- **输出目录**: results/rag_only

## 自定义评测指标

### 方式 1: 使用环境变量

#### 使用单个指标
```bash
EVALUATION_METRICS="exact_match" ./run_rag_env.sh
```

#### 使用多个指标
```bash
EVALUATION_METRICS="exact_match f1_score similarity" ./run_rag_env.sh
```

#### 使用所有推荐指标（问答任务）
```bash
EVALUATION_METRICS="exact_match f1_score similarity contains_answer" ./run_rag_env.sh
```

### 方式 2: 修改脚本中的默认值

编辑 `run_rag_env.sh` 第 14 行：
```bash
EVALUATION_METRICS="${EVALUATION_METRICS:-exact_match f1_score}"
```

## 可用的评测指标

| 指标名称 | 描述 | 适用场景 |
|---------|------|----------|
| `exact_match` | 精确匹配 | 答案需要完全相同 |
| `f1_score` | F1 分数 | 基于词重叠计算 |
| `similarity` | 相似度 | 语义相似度评测 |
| `contains_answer` | 包含答案 | 答案是否被包含在预测中 |
| `bleu_score` | BLEU 分数 | 机器翻译任务 |
| `rouge_score` | ROUGE 分数 | 文本摘要任务 |
| `numeric_match` | 数值匹配 | 数学计算任务 |
| `llm_judgement` | LLM 评判 | 复杂开放式问题 |

## 完整配置示例

### 示例 1: 自定义所有参数
```bash
DATA_PATH="data/my_benchmark.jsonl" \
NUM_ROLLOUTS=10 \
MODEL_NAME="gpt-4o-mini" \
MAX_TURNS=20 \
OUTPUT_DIR="results/my_experiment" \
MCP_SERVER_URL="http://192.168.1.100:8080" \
EVALUATION_METRICS="exact_match f1_score" \
./run_rag_env.sh
```

### 示例 2: 快速测试（单个指标）
```bash
NUM_ROLLOUTS=5 \
EVALUATION_METRICS="exact_match" \
./run_rag_env.sh
```

### 示例 3: 全面评测（多个指标）
```bash
NUM_ROLLOUTS=50 \
EVALUATION_METRICS="exact_match f1_score similarity contains_answer rouge_score" \
OUTPUT_DIR="results/comprehensive_eval" \
./run_rag_env.sh
```

### 示例 4: 数学任务评测
```bash
DATA_PATH="data/math_benchmark.jsonl" \
EVALUATION_METRICS="numeric_match exact_match contains_answer" \
OUTPUT_DIR="results/math_eval" \
./run_rag_env.sh
```

## 输出文件

运行完成后，在 `OUTPUT_DIR` 目录下会生成以下文件：

### 1. evaluation_scores.json
每个任务在所有指标下的详细评分：
```json
[
  {
    "task_id": "task_001",
    "predicted_answer": "Paris",
    "ground_truth": "Paris",
    "scores": {
      "exact_match": 1.0,
      "f1_score": 1.0,
      "similarity": 1.0,
      "contains_answer": 1.0
    },
    "is_correct": {
      "exact_match": true,
      "f1_score": true,
      "similarity": true,
      "contains_answer": true
    }
  }
]
```

### 2. evaluation_summary.json
所有指标的汇总统计：
```json
{
  "timestamp": "2025-12-06T10:30:00",
  "evaluation_metrics": ["exact_match", "f1_score", "similarity", "contains_answer"],
  "metrics_statistics": {
    "exact_match": {
      "total_items": 50,
      "successful_items": 38,
      "failed_items": 12,
      "average_score": 0.76,
      "success_rate": 0.76
    },
    "f1_score": {
      "total_items": 50,
      "successful_items": 42,
      "failed_items": 8,
      "average_score": 0.84,
      "success_rate": 0.84
    },
    ...
  }
}
```

### 3. trajectory.jsonl
每个任务的执行轨迹

### 4. worker_instance_map.json
Worker 状态映射

## 环境变量参考

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `DATA_PATH` | src/data/HotPotQA_demo.jsonl | 基准数据集路径 |
| `NUM_ROLLOUTS` | 50 | 并行 worker 数量 |
| `MODEL_NAME` | gpt-4.1-2025-04-14 | 使用的模型名称 |
| `MAX_TURNS` | 15 | 每个任务的最大轮次 |
| `OUTPUT_DIR` | results/rag_only | 结果输出目录 |
| `MCP_SERVER_URL` | http://localhost:8080 | MCP 服务器地址 |
| `EVALUATION_METRICS` | exact_match f1_score similarity contains_answer | 评测指标列表 |

## 推荐配置

### 问答任务（HotPotQA）
```bash
EVALUATION_METRICS="exact_match f1_score similarity contains_answer"
```

### 开放式生成任务
```bash
EVALUATION_METRICS="rouge_score bleu_score similarity"
```

### 数学计算任务
```bash
EVALUATION_METRICS="numeric_match exact_match contains_answer"
```

### 快速验证（单指标）
```bash
EVALUATION_METRICS="exact_match"
```

### 综合评测（多维度）
```bash
EVALUATION_METRICS="exact_match f1_score similarity contains_answer rouge_score"
```

## 性能提示

1. **指标数量与耗时**: 每增加一个指标，评测时间会相应增加，但不影响任务执行本身
2. **并行度设置**: `NUM_ROLLOUTS` 设置为 CPU 核心数的 1-2 倍可获得最佳性能
3. **LLM 评判**: `llm_judgement` 指标会调用大模型，显著增加评测时间和成本

## 故障排查

### 问题 1: 脚本没有执行权限
```bash
chmod +x run_rag_env.sh
```

### 问题 2: Python 模块找不到
确保在项目根目录运行：
```bash
cd /home/a1/sdb/lb/Mytest
./run_rag_env.sh
```

### 问题 3: MCP 服务器连接失败
检查 MCP 服务器是否运行：
```bash
curl http://localhost:8080/health
```

### 问题 4: 指标名称错误
参考上面的"可用的评测指标"表格，确保指标名称正确

## 查看运行日志

运行时会显示详细的日志输出：
```
==========================================
Running RAG-only Environment
==========================================
Data Path: src/data/HotPotQA_demo.jsonl
Num Rollouts: 50
Model: gpt-4.1-2025-04-14
Max Turns: 15
Output Dir: results/rag_only
MCP Server: http://localhost:8080
Evaluation Metrics: exact_match f1_score similarity contains_answer
==========================================

...

============================================================
Benchmark Evaluation Results
  Metrics: exact_match, f1_score, similarity, contains_answer
  Total Tasks: 50
  Successful Predictions: 48

  [exact_match]
    Total Items: 48
    Successful: 36
    Failed: 12
    Average Score: 0.7500
    Success Rate: 75.00%

  [f1_score]
    Total Items: 48
    Successful: 40
    Failed: 8
    Average Score: 0.8333
    Success Rate: 83.33%

  ...
============================================================
```

## 参考文档

- [MULTI_METRICS_USAGE.md](MULTI_METRICS_USAGE.md) - 多指标评测详细文档
- [src/run_parallel_rollout.py](src/run_parallel_rollout.py) - 主程序源码
- [src/benchmark/benchmark.py](src/benchmark/benchmark.py) - 评测指标实现
