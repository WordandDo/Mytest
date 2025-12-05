# HttpMCPRagEnv 快速开始

## 创建的新文件

1. **环境类**: [src/envs/http_mcp_rag_env.py](src/envs/http_mcp_rag_env.py)
   - 继承自 HttpMCPEnv 的 RAG 专用环境
   - 自动过滤只使用 RAG 资源
   - 使用定制化的 RAG 任务 System Prompt

2. **Bash 运行脚本**: [run_rag_env.sh](run_rag_env.sh)
   - 便捷的命令行脚本
   - 支持通过环境变量配置参数

3. **Python 示例**: [example_rag_env.py](example_rag_env.py)
   - 完整的 Python 示例代码
   - 展示如何编程式使用环境

4. **详细文档**: [RAG_ENV_USAGE.md](RAG_ENV_USAGE.md)
   - 完整的使用说明
   - 参数详解
   - 故障排查指南

## 三种运行方式

### 方式 1: 使用 Bash 脚本（最简单）

```bash
# 使用默认参数
./run_rag_env.sh

# 自定义参数
DATA_PATH=src/data/rag_demo.jsonl \
NUM_ROLLOUTS=3 \
MODEL_NAME=gpt-4.1-2025-04-14 \
./run_rag_env.sh
```

### 方式 2: 使用 Python 示例脚本

```bash
python example_rag_env.py
```

### 方式 3: 直接调用框架脚本

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

## 核心参数说明

### 必需参数

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `--data_path` | 测试数据文件 | `src/data/rag_demo.jsonl` |
| `--env_mode` | **必须设为 `http_mcp_rag`** | `http_mcp_rag` |

### 重要可选参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_rollouts` | 5 | 并行 Worker 数量 |
| `--mcp_server_url` | http://localhost:8080 | MCP Server 地址 |
| `--model_name` | gpt-4.1-2025-04-14 | 使用的 LLM 模型 |
| `--max_turns` | 15 | 每任务最大轮次 |
| `--output_dir` | results | 结果输出目录 |

### 环境变量（在 .env 中配置）

```bash
# OpenAI API
OPENAI_API_KEY=your-api-key-here
OPENAI_API_URL=https://api.openai.com/v1      # 可选
OPENAI_TIMEOUT=30                              # 可选
OPENAI_MAX_RETRIES=2                           # 可选

# 任务超时
TASK_EXECUTION_TIMEOUT=600
```

## 运行前检查清单

- [ ] MCP Server 正在运行并可访问
- [ ] `gateway_config.json` 包含 RAG 资源配置
- [ ] `.env` 文件包含 `OPENAI_API_KEY`
- [ ] 测试数据文件存在（如 `src/data/rag_demo.jsonl`）
- [ ] Python 环境安装了所有依赖

## 预期输出

运行成功后，会在输出目录生成：

```
results/rag_only/
├── trajectory.jsonl                    # 所有任务轨迹
├── worker_instance_map.json           # Worker 映射
└── gpt-4.1-2025-04-14/               # 按模型分组
    ├── task_001.json
    ├── task_002.json
    └── ...
```

控制台输出示例：

```
==========================================================
Starting Parallel Rollout Framework (MCP Native)
  Num Rollouts: 2
  Env Mode: http_mcp_rag
  Benchmark Items: 6
==========================================================
...
==========================================================
Benchmark Evaluation Results
  Metric: exact_match
  Total Items: 6
  Average Score: 0.8333
==========================================================
```

## 故障排查

**问题**: 找不到模块 `http_mcp_rag_env`

**解决**: 确认已更新 [src/envs/factory.py](src/envs/factory.py) 注册了新环境

**问题**: 连接不上 MCP Server

**解决**:
```bash
# 检查 MCP Server 状态
curl http://localhost:8080/health
```

**问题**: OpenAI API 错误

**解决**: 检查 `OPENAI_API_KEY` 和 `OPENAI_API_URL` 设置

## 更多信息

详细文档请参考：[RAG_ENV_USAGE.md](RAG_ENV_USAGE.md)
