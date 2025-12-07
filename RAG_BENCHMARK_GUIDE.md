# RAG 环境测评使用指南

本文档介绍如何使用 `http_mcp_rag_env.py` 环境进行测评，支持 `exact_match` 和 `f1_score` 两种测评方案。

## 快速开始

### 方法 1: 使用 Bash 脚本（推荐）

```bash
# 使用默认配置运行
./run_rag_benchmark.sh

# 自定义配置
DATA_PATH=src/data/HotPotQA.jsonl \
NUM_ROLLOUTS=10 \
OUTPUT_DIR=results/my_test \
./run_rag_benchmark.sh
```

### 方法 2: 使用 Python 脚本

```bash
# 使用默认配置
python run_rag_test.py

# 使用环境变量自定义
DATA_PATH=src/data/HotPotQA.jsonl \
NUM_ROLLOUTS=10 \
MODEL_NAME=gpt-4.1-2025-04-14 \
python run_rag_test.py
```

### 方法 3: 直接使用 run_parallel_rollout.py

```bash
python src/run_parallel_rollout.py \
    --data_path src/data/rag_demo.jsonl \
    --num_rollouts 5 \
    --env_mode http_mcp_rag \
    --output_dir results/rag_test \
    --model_name gpt-4.1-2025-04-14 \
    --max_turns 15 \
    --evaluation_metric exact_match f1_score
```

## 配置说明

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `DATA_PATH` | 测试数据文件路径 | `src/data/rag_demo.jsonl` |
| `NUM_ROLLOUTS` | 并行 Worker 数量 | `5` |
| `OUTPUT_DIR` | 结果输出目录 | `results/rag_test_<timestamp>` |
| `MODEL_NAME` | LLM 模型名称 | `gpt-4.1-2025-04-14` |
| `MAX_TURNS` | 每个任务最大轮次 | `15` |
| `MCP_SERVER_URL` | MCP 服务器地址 | `http://localhost:8080` |
| `RESOURCE_API_URL` | 资源 API 地址 | `http://localhost:8000` |
| `GATEWAY_CONFIG_PATH` | Gateway 配置文件 | `gateway_config.json` |

### 命令行参数（使用 run_parallel_rollout.py）

```bash
--data_path          # 测试数据路径
--num_rollouts       # 并行 Worker 数量
--env_mode           # 环境模式 (http_mcp_rag)
--output_dir         # 输出目录
--model_name         # 模型名称
--max_turns          # 最大轮次
--evaluation_metric  # 测评指标（可指定多个）
--mcp_server_url     # MCP 服务器 URL
--resource_api_url   # 资源 API URL
```

## 测评指标说明

本测评支持两种标准指标：

### 1. Exact Match (精确匹配)
- **说明**: 预测答案经过标准化处理后与标准答案完全匹配
- **标准化处理**:
  - 转换为小写
  - 移除冠词 (a, an, the)
  - 移除标点符号
  - 修复多余空白
- **得分**: 0 或 1

### 2. F1 Score (F1 分数)
- **说明**: 基于词袋模型计算 Precision 和 Recall
- **计算方式**:
  - Precision = 重叠词数 / 预测词数
  - Recall = 重叠词数 / 标准答案词数
  - F1 = 2 × (Precision × Recall) / (Precision + Recall)
- **特殊处理**: 对于 yes/no/noanswer 类型的答案，如果不完全匹配则给 0 分
- **得分**: 0.0 到 1.0 之间的连续值

## 测试数据格式

测试数据支持 JSON 和 JSONL 格式，每个测试项包含：

```json
{
  "id": "test_001",
  "question": "问题内容",
  "answer": "标准答案"
}
```

### 可用测试数据集

- `src/data/rag_demo.jsonl` - RAG 示例数据（5条）
- `src/data/HotPotQA.jsonl` - HotPotQA 完整数据集
- `src/data/HotPotQA_demo.jsonl` - HotPotQA 示例数据
- `src/data/bamboogle.json` - Bamboogle 数据集

## 输出文件说明

测评完成后，在输出目录中会生成以下文件：

### 1. trajectory.jsonl
包含每个任务的完整执行轨迹：
```json
{
  "task_id": "test_001",
  "success": true,
  "answer": "预测答案",
  "messages": [...],  // 完整的对话历史
  "steps": [...],     // 执行步骤
  "duration": 12.5
}
```

### 2. evaluation_scores.json
详细的评分结果：
```json
[
  {
    "task_id": "test_001",
    "predicted_answer": "预测答案",
    "ground_truth": "标准答案",
    "scores": {
      "exact_match": 1.0,
      "f1_score": 0.8571
    },
    "is_correct": {
      "exact_match": true,
      "f1_score": true
    }
  }
]
```

### 3. evaluation_summary.json
汇总统计信息：
```json
{
  "timestamp": "2025-12-07T10:30:00",
  "evaluation_metrics": ["exact_match", "f1_score"],
  "metrics_statistics": {
    "exact_match": {
      "total_items": 50,
      "successful_items": 35,
      "failed_items": 15,
      "average_score": 0.70,
      "success_rate": 0.70
    },
    "f1_score": {
      "total_items": 50,
      "successful_items": 40,
      "failed_items": 10,
      "average_score": 0.78,
      "success_rate": 0.80
    }
  },
  "execution_time": {
    "total_seconds": 125.5,
    "formatted": "00:02:05",
    "start_time": "2025-12-07 10:28:00",
    "end_time": "2025-12-07 10:30:05"
  }
}
```

### 4. worker_instance_map.json
Worker 和资源的映射关系（调试用）

## 前置要求

### 1. 启用 RAG 资源

确保 `deployment_config.json` 中 RAG 资源已启用：

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

### 2. 启动资源服务器

```bash
# 启动资源网关服务
python -m utils.resource_pools.gateway_server
```

### 3. 配置环境变量

创建 `.env` 文件并配置必要的环境变量：

```bash
OPENAI_API_KEY=your_api_key
OPENAI_API_BASE=https://api.openai.com/v1
```

## 示例

### 示例 1: 快速测试（5个样本）

```bash
./run_rag_benchmark.sh
```

### 示例 2: 完整测评（完整 HotPotQA 数据集）

```bash
DATA_PATH=src/data/HotPotQA.jsonl \
NUM_ROLLOUTS=20 \
OUTPUT_DIR=results/hotpotqa_full \
./run_rag_benchmark.sh
```

### 示例 3: 自定义模型和参数

```bash
python run_rag_test.py
```

或者通过环境变量：

```bash
export DATA_PATH=src/data/bamboogle.json
export NUM_ROLLOUTS=10
export MODEL_NAME=gpt-4.1-2025-04-14
export MAX_TURNS=20
python run_rag_test.py
```

## 常见问题

### Q1: 如何使用不同的 RAG 索引？

修改 `deployment_config.json` 中的 RAG 配置：

```json
"rag": {
  "config": {
    "rag_index_path": "/path/to/your/index",
    "rag_model_name": "your-model-name"
  }
}
```

### Q2: 如何调整并行度？

通过 `NUM_ROLLOUTS` 参数调整：

```bash
NUM_ROLLOUTS=20 ./run_rag_benchmark.sh
```

### Q3: 如何只使用一个测评指标？

修改脚本中的评测指标配置，或使用命令行：

```bash
python src/run_parallel_rollout.py \
    --evaluation_metric exact_match \
    ...
```

### Q4: 如何查看详细的执行日志？

查看输出目录中的 `trajectory.jsonl` 文件，包含完整的执行过程。

## 性能优化建议

1. **调整并行度**: 根据可用 CPU 核心数调整 `NUM_ROLLOUTS`
2. **使用 GPU**: 在 `deployment_config.json` 中启用 `use_gpu_index`
3. **调整 top_k**: 减少检索的文档数量可以提高速度
4. **批量处理**: 对大数据集使用更高的并行度

## 注意事项

1. 确保 RAG 索引路径正确且可访问
2. 确保有足够的内存加载索引（特别是大索引）
3. 首次运行可能需要较长时间加载模型和索引
4. 建议先用小数据集测试配置是否正确

## 技术支持

如有问题，请检查：
1. 日志输出中的错误信息
2. `deployment_config.json` 配置是否正确
3. 资源服务器是否正常运行
4. 环境变量是否正确设置
