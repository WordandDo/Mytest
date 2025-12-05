# HttpMCPRagEnv 使用文档

## 概述

`HttpMCPRagEnv` 是一个专门用于 RAG（检索增强生成）任务的运行环境，继承自 `HttpMCPEnv`。该环境自动过滤只使用 RAG 类型的资源，并应用专门针对信息检索任务优化的 System Prompt。

## 特性

1. **RAG 专用**: 自动过滤 gateway 配置，只加载 `resource_type == "rag"` 的模块
2. **定制化 Prompt**: 使用专门针对工具检索和信息查询优化的系统提示
3. **完全继承**: 继承 `HttpMCPEnv` 的所有功能（MCP 协议、资源管理、Agent 执行循环）
4. **灵活配置**: 支持所有 `HttpMCPEnv` 的配置参数

## 文件位置

- 环境类: [src/envs/http_mcp_rag_env.py](src/envs/http_mcp_rag_env.py)
- 运行脚本: [run_rag_env.sh](run_rag_env.sh)
- 示例数据: [src/data/rag_demo.jsonl](src/data/rag_demo.jsonl)

## 快速开始

### 方法 1: 使用便捷脚本

```bash
# 基本运行（使用默认参数）
./run_rag_env.sh

# 自定义参数
DATA_PATH=src/data/rag_demo.jsonl \
NUM_ROLLOUTS=3 \
MODEL_NAME=gpt-4.1-2025-04-14 \
MAX_TURNS=20 \
OUTPUT_DIR=results/my_rag_test \
MCP_SERVER_URL=http://localhost:8080 \
./run_rag_env.sh
```

### 方法 2: 直接使用 Python 脚本

```bash
python src/run_parallel_rollout.py \
    --data_path src/data/rag_demo.jsonl \
    --num_rollouts 2 \
    --env_mode http_mcp_rag \
    --output_dir results/rag_only \
    --mcp_server_url http://localhost:8080 \
    --model_name gpt-4.1-2025-04-14 \
    --max_turns 15
```

## 脚本参数说明

### 必需参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--data_path` | 基准测试数据文件路径 | `src/data/rag_demo.jsonl` |
| `--env_mode` | 环境模式（使用 RAG 环境需设为 `http_mcp_rag`） | `http_mcp_rag` |

### 可选参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_rollouts` | 5 | 并行 Worker 数量 |
| `--output_dir` | results | 结果输出目录 |
| `--mcp_server_url` | http://localhost:8080 | MCP Server 地址 |
| `--model_name` | gpt-4.1-2025-04-14 | LLM 模型名称 |
| `--max_turns` | 15 | 每个任务最大轮次 |

### 环境变量

这些变量可以在 `.env` 文件中设置或通过命令行导出：

```bash
# OpenAI API 配置
export OPENAI_API_KEY="your-api-key"
export OPENAI_API_URL="https://api.openai.com/v1"  # 可选
export OPENAI_TIMEOUT="30"                          # 可选
export OPENAI_MAX_RETRIES="2"                       # 可选

# 任务执行超时（秒）
export TASK_EXECUTION_TIMEOUT="600"
```

## Gateway 配置

确保您的 `gateway_config.json` 包含 RAG 资源配置：

```json
{
  "modules": [
    {
      "resource_type": "rag",
      "tools": ["rag_search", "rag_retrieve"],
      "config": {
        // RAG 配置参数
      }
    }
  ]
}
```

**注意**: `HttpMCPRagEnv` 会自动过滤掉所有非 RAG 类型的资源，即使它们在配置文件中存在。

## 数据格式

基准测试数据文件应该是 JSON 或 JSONL 格式，每个条目包含：

```json
{
  "id": "task_001",
  "question": "问题描述",
  "answer": "标准答案",
  "metadata": {
    // 可选的元数据
  }
}
```

示例（JSONL 格式）：

```jsonl
{"id": "5a8b57f25542995d1e6f1371", "question": "Were Scott Derrickson and Ed Wood of the same nationality?", "answer": "yes"}
{"id": "5a8c7595554299585d9e36b6", "question": "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?", "answer": "Chief of Protocol"}
```

## 输出结果

运行完成后，结果将保存在指定的输出目录中：

```
results/rag_only/
├── trajectory.jsonl              # 所有任务的执行轨迹
├── worker_instance_map.json      # Worker 资源映射（调试用）
└── gpt-4.1-2025-04-14/          # 按模型分组的日志
    ├── task_001.json
    ├── task_002.json
    └── ...
```

每个任务的 JSON 文件包含：

```json
{
  "meta": {
    "task_id": "task_001",
    "model_name": "gpt-4.1-2025-04-14",
    "timestamp": "2025-12-05T10:30:00",
    "output_file": "results/rag_only/gpt-4.1-2025-04-14/task_001.json"
  },
  "task": {
    "question": "问题描述",
    "status": "success",
    "final_answer": "模型的答案",
    "total_turns": 5
  },
  "raw_result": {
    "task_id": "task_001",
    "question": "问题描述",
    "answer": "模型的答案",
    "messages": [...],  // 完整的对话历史
    "success": true,
    "error": null
  }
}
```

## System Prompt

`HttpMCPRagEnv` 使用以下定制化 System Prompt：

```
You are a helpful assistant. You need to use tools to solve the problem.
You must use tool to retrieve information to answer and verify. Don't answer by your own knowledge.

## Tool Usage Strategy
1. Break complex problems into logical steps
2. Use ONE tool at a time to gather information
3. Verify findings through different approaches when possible
4. You are encouraged to perform multiple tool-use to get the final answer. If some tool call doesn't output useful response, try other one.
5. You need to use the information you retrieve carefully and accurately, and avoid making incorrect associations or assumptions.
6. After several (more than 10) tool-use, if you still can't get the final answer, you can answer by your own knowledge.

## Answer Strategy
The final answer only contains the short answer to the question (few words), no other words like reasoning content.
```

## 编程接口

如果需要在代码中使用此环境：

```python
from src.envs.http_mcp_rag_env import HttpMCPRagEnv
from src.benchmark.benchmark import Benchmark

# 创建环境实例
env = HttpMCPRagEnv(
    model_name="gpt-4.1-2025-04-14",
    mcp_server_url="http://localhost:8080",
    gateway_config_path="gateway_config.json",
    worker_id="worker-1"
)

# 启动环境
env.env_start()

# 分配资源
env.allocate_resource("worker-1", resource_init_data={})

# 运行任务
task = {
    "id": "test_001",
    "question": "Your question here?",
    "answer": "Expected answer"
}

agent_config = {
    "model_name": "gpt-4.1-2025-04-14",
    "max_turns": 15,
    "max_retries": 2
}

result = env.run_task(task, agent_config, logger)

# 释放资源
env.release_resource("worker-1")

# 关闭环境
env.env_close()
```

## 高级配置

### 自定义 System Prompt

如果需要修改 System Prompt，编辑 [src/envs/http_mcp_rag_env.py](src/envs/http_mcp_rag_env.py) 中的 `SYSTEM_PROMPT_GENERIC` 变量。

### 资源过滤逻辑

默认情况下，环境只保留 `resource_type == "rag"` 的模块。如果需要修改过滤逻辑，可以重写 `_load_gateway_config()` 方法。

### 观察黑名单

可以在任务的 metadata 中配置观察黑名单：

```json
{
  "id": "task_001",
  "question": "问题",
  "answer": "答案",
  "metadata": {
    "observation_blacklist": ["vm_pyautogui"],
    "observation_content_blacklist": {
      "vm_computer_13": ["accessibility_tree"]
    }
  }
}
```

## 故障排查

### 问题 1: 找不到 RAG 资源

**症状**: 日志显示 "No allocatable resources found"

**解决方案**:
- 检查 `gateway_config.json` 中是否有 `resource_type: "rag"` 的模块
- 确认 MCP Server 正在运行并且可以访问

### 问题 2: 连接 MCP Server 失败

**症状**: "Failed to connect to MCP Server"

**解决方案**:
- 确认 MCP Server 正在运行: `curl http://localhost:8080/health`
- 检查 `--mcp_server_url` 参数是否正确
- 查看防火墙设置

### 问题 3: OpenAI API 错误

**症状**: "OpenAI API returned empty response" 或超时

**解决方案**:
- 检查 `OPENAI_API_KEY` 是否正确设置
- 增加 `OPENAI_TIMEOUT` 环境变量
- 验证 `OPENAI_API_URL` 设置（如果使用自定义端点）

### 问题 4: 任务超时

**症状**: "Task execution timeout"

**解决方案**:
- 增加 `TASK_EXECUTION_TIMEOUT` 环境变量（默认 600 秒）
- 减少 `--max_turns` 参数
- 优化问题复杂度

## 与 HttpMCPEnv 的对比

| 特性 | HttpMCPEnv | HttpMCPRagEnv |
|------|-----------|---------------|
| 资源类型 | 所有配置的资源 | 仅 RAG 资源 |
| System Prompt | 通用 Agent Prompt | RAG 专用 Prompt |
| 模式标识 | `http_mcp` | `http_mcp_rag` |
| 使用场景 | 多模态任务、GUI 操作等 | 知识检索、问答任务 |

## 性能优化建议

1. **并行度**: 根据 MCP Server 容量调整 `--num_rollouts`
2. **轮次限制**: 合理设置 `--max_turns`，避免无效循环
3. **资源预热**: 在批量任务前先运行小规模测试
4. **日志管理**: 定期清理输出目录，避免磁盘占满

## 更多信息

- MCP 协议文档: 查看 `src/mcp_server/` 目录
- 基准测试框架: [src/benchmark/benchmark.py](src/benchmark/benchmark.py)
- 并行执行框架: [src/run_parallel_rollout.py](src/run_parallel_rollout.py)
- 父类环境: [src/envs/http_mcp_env.py](src/envs/http_mcp_env.py)
