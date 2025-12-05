# 多指标评测使用指南

本文档介绍如何在 `run_parallel_rollout.py` 中使用多指标评测功能。

## 支持的评分指标

系统支持以下 8 种评分指标：

1. **exact_match** - 精确匹配（完全相同才得分）
2. **f1_score** - F1 分数（基于词重叠计算）
3. **bleu_score** - BLEU 分数（机器翻译常用）
4. **rouge_score** - ROUGE 分数（文本摘要常用）
5. **similarity** - 相似度（基于文本相似性）
6. **contains_answer** - 包含答案（答案是否被包含）
7. **numeric_match** - 数值匹配（数值是否相等）
8. **llm_judgement** - LLM 评判（使用大模型评判）

## 启动方式

### 方法 1：命令行参数（需要修改脚本）

如果需要通过命令行参数指定多个评分指标，需要先修改脚本添加参数解析：

#### 修改后的命令行使用（单个指标）
```bash
python src/run_parallel_rollout.py \
    --data_path data/benchmark.jsonl \
    --num_rollouts 5 \
    --env_mode http_mcp \
    --output_dir results \
    --evaluation_metric exact_match
```

#### 修改后的命令行使用（多个指标）
```bash
python src/run_parallel_rollout.py \
    --data_path data/benchmark.jsonl \
    --num_rollouts 5 \
    --env_mode http_mcp \
    --output_dir results \
    --evaluation_metric exact_match f1_score similarity
```

### 方法 2：Python 代码调用（推荐）

#### 示例 1：使用单个指标（向后兼容）

```python
from src.run_parallel_rollout import run_parallel_rollout, ParallelRolloutConfig
from benchmark import Benchmark

# 加载基准数据
benchmark = Benchmark(data_path="data/benchmark.jsonl")

# 配置
config = ParallelRolloutConfig(
    num_rollouts=5,
    env_mode="http_mcp",
    output_dir="results",
    env_kwargs={
        "observation_type": "screenshot_a11y_tree",
        "mcp_server_url": "http://localhost:8080",
        "resource_api_url": "http://localhost:8000",
    },
    agent_config_dict={
        "model_name": "gpt-4.1-2025-04-14",
        "evaluation_metric": "exact_match",  # 单个指标
        "max_turns": 15,
        "max_retries": 2
    }
)

# 运行测评
results = run_parallel_rollout(config, benchmark)
```

#### 示例 2：使用多个指标

```python
from src.run_parallel_rollout import run_parallel_rollout, ParallelRolloutConfig
from benchmark import Benchmark

# 加载基准数据
benchmark = Benchmark(data_path="data/benchmark.jsonl")

# 配置 - 使用多个评分指标
config = ParallelRolloutConfig(
    num_rollouts=5,
    env_mode="http_mcp",
    output_dir="results",
    env_kwargs={
        "observation_type": "screenshot_a11y_tree",
        "mcp_server_url": "http://localhost:8080",
        "resource_api_url": "http://localhost:8000",
    },
    agent_config_dict={
        "model_name": "gpt-4.1-2025-04-14",
        # 使用列表指定多个评分指标
        "evaluation_metric": [
            "exact_match",
            "f1_score",
            "similarity",
            "contains_answer"
        ],
        "max_turns": 15,
        "max_retries": 2
    }
)

# 运行测评
results = run_parallel_rollout(config, benchmark)

# 查看结果
print("Metrics Statistics:")
for metric, stats in results["metrics_statistics"].items():
    print(f"\n{metric}:")
    print(f"  Average Score: {stats['average_score']:.4f}")
    print(f"  Success Rate: {stats['success_rate']:.2%}")
```

#### 示例 3：针对不同任务类型使用不同指标组合

```python
# 问答任务 - 使用精确匹配和包含答案
qa_config = ParallelRolloutConfig(
    agent_config_dict={
        "evaluation_metric": ["exact_match", "contains_answer", "f1_score"]
    }
)

# 摘要任务 - 使用 ROUGE 和相似度
summary_config = ParallelRolloutConfig(
    agent_config_dict={
        "evaluation_metric": ["rouge_score", "similarity", "bleu_score"]
    }
)

# 数学计算任务 - 使用数值匹配和包含答案
math_config = ParallelRolloutConfig(
    agent_config_dict={
        "evaluation_metric": ["numeric_match", "contains_answer", "exact_match"]
    }
)
```

## 输出文件格式

运行后会在 `output_dir` 目录下生成以下文件：

### 1. evaluation_scores.json

包含每个任务在所有指标下的详细评分：

```json
[
  {
    "task_id": "task_001",
    "predicted_answer": "The capital of France is Paris",
    "ground_truth": "Paris",
    "scores": {
      "exact_match": 0.0,
      "f1_score": 0.67,
      "similarity": 0.85,
      "contains_answer": 1.0
    },
    "is_correct": {
      "exact_match": false,
      "f1_score": true,
      "similarity": true,
      "contains_answer": true
    }
  },
  {
    "task_id": "task_002",
    "predicted_answer": "42",
    "ground_truth": "42",
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

包含所有指标的汇总统计：

```json
{
  "timestamp": "2025-12-06T10:30:00.123456",
  "evaluation_metrics": [
    "exact_match",
    "f1_score",
    "similarity",
    "contains_answer"
  ],
  "metrics_statistics": {
    "exact_match": {
      "total_items": 20,
      "successful_items": 15,
      "failed_items": 5,
      "average_score": 0.75,
      "success_rate": 0.75
    },
    "f1_score": {
      "total_items": 20,
      "successful_items": 18,
      "failed_items": 2,
      "average_score": 0.82,
      "success_rate": 0.90
    },
    "similarity": {
      "total_items": 20,
      "successful_items": 19,
      "failed_items": 1,
      "average_score": 0.88,
      "success_rate": 0.95
    },
    "contains_answer": {
      "total_items": 20,
      "successful_items": 19,
      "failed_items": 1,
      "average_score": 0.95,
      "success_rate": 0.95
    }
  },
  "execution_time": {
    "total_seconds": 1234.56,
    "formatted": "00:20:34",
    "start_time": "2025-12-06 10:10:00",
    "end_time": "2025-12-06 10:30:34"
  },
  "configuration": {
    "num_rollouts": 5,
    "env_mode": "http_mcp",
    "model_name": "gpt-4.1-2025-04-14",
    "max_turns": 15
  },
  "tasks_summary": {
    "total_tasks": 20,
    "successful_predictions": 20,
    "failed_predictions": 0
  }
}
```

### 3. trajectory.jsonl

执行轨迹（不受多指标影响）

### 4. worker_instance_map.json

Worker 状态映射（不受多指标影响）

## 日志输出示例

运行时会在日志中显示每个指标的详细统计：

```
2025-12-06 10:30:00 - INFO - Evaluating with 4 metric(s): exact_match, f1_score, similarity, contains_answer
2025-12-06 10:30:01 - INFO -   Computing metric: exact_match...
2025-12-06 10:30:02 - INFO -   Computing metric: f1_score...
2025-12-06 10:30:03 - INFO -   Computing metric: similarity...
2025-12-06 10:30:04 - INFO -   Computing metric: contains_answer...
2025-12-06 10:30:05 - INFO - ============================================================
2025-12-06 10:30:05 - INFO - Benchmark Evaluation Results
2025-12-06 10:30:05 - INFO -   Metrics: exact_match, f1_score, similarity, contains_answer
2025-12-06 10:30:05 - INFO -   Total Tasks: 20
2025-12-06 10:30:05 - INFO -   Successful Predictions: 20
2025-12-06 10:30:05 - INFO -
2025-12-06 10:30:05 - INFO -   [exact_match]
2025-12-06 10:30:05 - INFO -     Total Items: 20
2025-12-06 10:30:05 - INFO -     Successful: 15
2025-12-06 10:30:05 - INFO -     Failed: 5
2025-12-06 10:30:05 - INFO -     Average Score: 0.7500
2025-12-06 10:30:05 - INFO -     Success Rate: 75.00%
2025-12-06 10:30:05 - INFO -   [f1_score]
2025-12-06 10:30:05 - INFO -     Total Items: 20
2025-12-06 10:30:05 - INFO -     Successful: 18
2025-12-06 10:30:05 - INFO -     Failed: 2
2025-12-06 10:30:05 - INFO -     Average Score: 0.8200
2025-12-06 10:30:05 - INFO -     Success Rate: 90.00%
2025-12-06 10:30:05 - INFO - ============================================================
```

## 常见问题

### Q1: 如何选择合适的评分指标？

**建议：**
- **问答任务**: `["exact_match", "f1_score", "contains_answer"]`
- **生成任务**: `["rouge_score", "bleu_score", "similarity"]`
- **数学计算**: `["numeric_match", "exact_match"]`
- **综合评测**: `["exact_match", "f1_score", "similarity", "contains_answer"]`

### Q2: 多个指标会影响性能吗？

会的。每增加一个指标，评测时间会相应增加。但由于评测是在任务执行完成后进行的，不会影响任务执行本身的性能。

### Q3: 可以自定义评分指标吗？

目前支持 8 种内置指标。如需自定义指标，需要在 `src/benchmark/benchmark.py` 中添加相应的评分函数。

### Q4: 向后兼容性如何？

完全向后兼容。如果仍然使用单个字符串指定指标（如 `"exact_match"`），系统会自动转换为列表格式处理。

## 修改命令行参数支持（可选）

如果需要通过命令行直接指定多个指标，可以修改 `run_parallel_rollout.py` 的参数解析部分：

```python
# 在 if __name__ == "__main__": 部分添加
parser.add_argument(
    "--evaluation_metric",
    type=str,
    nargs='+',  # 接收多个参数
    default=["exact_match"],
    help="Evaluation metric(s) to use. Can specify multiple: --evaluation_metric exact_match f1_score"
)

# 修改 agent_config_dict
agent_config_dict={
    "model_name": args.model_name,
    "evaluation_metric": args.evaluation_metric if len(args.evaluation_metric) > 1 else args.evaluation_metric[0],
    "max_turns": args.max_turns,
    "max_retries": 2
}
```

## 总结

- ✅ 支持单个或多个评分指标
- ✅ 完全向后兼容
- ✅ 详细的评分记录和统计
- ✅ 灵活的配置方式
- ✅ 清晰的日志输出
