# RAG Benchmark Implementation Summary

## 概述

本次修改为 RAG 系统添加了四种不同的 Prompt 模板，并创建了对应的基准测试脚本，以支持不同的检索策略对比实验。

## 新增文件

### 1. Benchmark 脚本
- ✅ `benchmark_dense.sh` - Dense-only RAG 基准测试
- ✅ `benchmark_sparse.sh` - Sparse-only RAG 基准测试
- ✅ `benchmark_hybrid.sh` - Hybrid RAG 基准测试
- ✅ `benchmark_no_tool.sh` - 无工具基线测试（纯 LLM）
- ✅ `run_all_benchmarks.sh` - 运行所有基准测试的便捷脚本

### 2. Gateway 配置文件
- ✅ `gateway_config_rag_dense_only.json` - Dense 检索工具配置
- ✅ `gateway_config_rag_sparse_only.json` - Sparse 检索工具配置
- ✅ `gateway_config_rag_hybrid.json` - Hybrid 检索工具配置
- ✅ `gateway_config_rag_no_tool.json` - 无 RAG 工具配置

### 3. 文档
- ✅ `BENCHMARK_GUIDE.md` - 基准测试使用指南
- ✅ `BENCHMARK_COMPARISON.md` - 详细的配置对比和实验设计建议
- ✅ `IMPLEMENTATION_SUMMARY.md` - 本文档

## 修改的文件

### 1. `src/envs/http_mcp_rag_env.py`

**新增内容**：
- 添加了 System Prompts 章节分隔符
- 新增三个 Prompt 模板：
  - `SYSTEM_PROMPT_NO_TOOLS` - 不使用工具的版本
  - `SYSTEM_PROMPT_SPARSE` - Sparse 检索专用版本
  - `SYSTEM_PROMPT_HYBRID` - 混合检索增强版本

**修改内容**：
- `__init__` 方法：
  - 添加 `self.prompt_type` 实例变量以存储 prompt 类型
  - 从 `kwargs` 中提取 `prompt_type` 参数

- `get_system_prompt` 方法：
  - 更新参数文档，说明支持 `prompt_type`
  - 优先使用实例的 `prompt_type`，然后是 `kwargs` 中的值
  - 添加对四种 prompt 类型的分支处理
  - 对 `no_tool` 类型跳过添加工具描述

### 2. `src/run_parallel_rollout.py`

**新增内容**：
- 添加 `--prompt_type` 命令行参数
  - 类型：字符串
  - 默认值：`generic`
  - 可选值：`generic`, `no_tool`, `sparse`, `hybrid`
  - 帮助文本说明用途

**修改内容**：
- `env_kwargs` 字典：添加 `"prompt_type": args.prompt_type`

### 3. `run_rag_benchmark.sh`

**新增内容**：
- 添加 `PROMPT_TYPE` 环境变量（默认值：`generic`）
- 在配置信息输出中添加 "Prompt Type" 行
- 在 Python 命令中添加 `--prompt_type "$PROMPT_TYPE"` 参数

### 4. 所有 Benchmark 脚本

**统一格式**：
每个脚本都遵循相同的结构：
1. 清理旧的 Gateway（端口 8080）
2. 启动对应的 Gateway
3. 等待 Gateway 启动（5秒）
4. 设置环境变量（OUTPUT_DIR, DATA_PATH, NUM_ROLLOUTS, GATEWAY_CONFIG_PATH, PROMPT_TYPE）
5. 调用 `run_rag_benchmark.sh`
6. 清理并关闭 Gateway

## Prompt 类型详解

### 1. `generic` (通用版本)
**用途**：适用于任何 RAG 检索工具

**特点**：
- 通用的工具使用指导
- 不区分检索方法
- 基础的策略建议

**适用场景**：Dense-only 配置

---

### 2. `no_tool` (无工具版本)
**用途**：纯 LLM 基线测试

**特点**：
- 明确指示不使用任何工具
- 仅依赖模型内部知识
- 不添加工具描述到 prompt

**适用场景**：建立基线以衡量 RAG 的提升效果

---

### 3. `sparse` (Sparse 专用版本)
**用途**：优化关键词检索策略

**特点**：
- 强调 BM25 的关键词匹配特性
- 指导如何构建关键词查询
- 适合查找特定实体、术语、ID

**适用场景**：Sparse-only 配置（BM25）

**关键指导**：
- 使用精确关键词
- 避免模糊或概念性查询
- 适用于实体、术语、ID、精确短语

---

### 4. `hybrid` (混合增强版本)
**用途**：智能选择检索方法

**特点**：
- 明确区分 Sparse 和 Dense 的优势
- 提供选择工具的决策指南
- 支持多轮尝试和策略调整

**适用场景**：Hybrid 配置（同时提供 BM25 和 E5）

**Sparse 使用场景**：
- 精确名称、ID、代码、特定数字
- 罕见技术术语或行话
- 验证短语的精确存在
- Dense 检索返回幻觉或不相关结果时

**Dense 使用场景**：
- 概念、摘要、解释
- 不知道确切关键词但知道含义
- 广泛探索主题

---

## 技术实现细节

### Prompt 传递流程

```
用户脚本 (benchmark_*.sh)
  ↓ 设置 PROMPT_TYPE 环境变量
run_rag_benchmark.sh
  ↓ 通过 --prompt_type 参数传递
src/run_parallel_rollout.py
  ↓ 解析为 args.prompt_type
  ↓ 放入 env_kwargs
HttpMCPRagEnv.__init__()
  ↓ 存储为 self.prompt_type
HttpMCPRagEnv.get_system_prompt()
  ↓ 根据 prompt_type 选择对应的 SYSTEM_PROMPT_*
返回最终 System Prompt
```

### 配置文件结构

所有 Gateway 配置文件都包含：
- `system` 资源类型（必需，提供系统工具）
- `rag_hybrid` 资源类型（可选，提供 RAG 工具）
  - `tool_groups`：根据配置包含不同的工具组
    - Dense only: `["rag_query"]`
    - Sparse only: `["rag_query_sparse"]`
    - Hybrid: `["rag_query", "rag_query_sparse"]`
    - No tool: 无 RAG 工具组

### 端口配置

所有 Gateway 统一使用端口 8080：
- 每个脚本启动前会清理该端口的旧进程
- 使用 `lsof -ti:8080 | xargs kill -9 2>/dev/null`
- 如果需要使用其他端口，修改脚本中的 `--port` 参数

---

## 使用方法

### 运行单个基准测试

```bash
# Dense-only
./benchmark_dense.sh

# Sparse-only
./benchmark_sparse.sh

# Hybrid
./benchmark_hybrid.sh

# No Tool (baseline)
./benchmark_no_tool.sh
```

### 运行所有基准测试

```bash
./run_all_benchmarks.sh
```

### 自定义参数

在脚本中修改环境变量：
```bash
export OUTPUT_DIR="results/my_test"
export DATA_PATH="src/data/my_dataset.json"
export NUM_ROLLOUTS=20
export PROMPT_TYPE="hybrid"
```

或直接使用 `run_rag_benchmark.sh`：
```bash
export OUTPUT_DIR="results/custom"
export PROMPT_TYPE="sparse"
./run_rag_benchmark.sh
```

---

## 输出结构

```
results/
├── benchmark_dense_only/       # Dense 检索结果
├── benchmark_sparse_only/      # Sparse 检索结果
├── benchmark_hybrid/           # Hybrid 检索结果
└── benchmark_no_tool/          # 无工具基线结果
```

每个目录包含：
- 详细的 rollout 日志
- 评估指标（exact_match, f1_score）
- 任务完成统计

---

## 评估指标

所有基准测试默认使用两种评估指标：
1. **exact_match**：精确字符串匹配
2. **f1_score**：Token 级别 F1 分数

可在 `run_rag_benchmark.sh` 中修改 `--evaluation_metric` 参数来添加或删除指标。

---

## 实验建议

### 基础对比实验
1. 先运行 `benchmark_no_tool.sh` 建立基线
2. 运行单方法测试（Dense, Sparse）
3. 运行 Hybrid 测试
4. 对比性能提升

### 问题类型分析
根据问题类型分析不同方法的表现：
- **事实性问题**：Sparse 可能表现更好
- **概念性问题**：Dense 可能表现更好
- **混合问题**：Hybrid 应该表现最佳

### 数据集考虑
- **Bamboogle**：包含事实和概念问题的混合
- **自定义数据集**：修改 `DATA_PATH` 使用自己的数据

---

## 故障排查

### Gateway 启动失败
```bash
# 检查端口占用
lsof -i:8080

# 手动清理
kill -9 <PID>
```

### Prompt Type 不生效
1. 检查 `PROMPT_TYPE` 环境变量是否设置
2. 确认 `run_parallel_rollout.py` 接收到 `--prompt_type` 参数
3. 验证 `HttpMCPRagEnv.get_system_prompt()` 使用正确的 prompt

### 工具不可用
1. 检查 Gateway 配置文件中的 `tool_groups`
2. 确认 MCP Server 正确初始化
3. 查看环境日志中的工具注册信息

---

## 兼容性

### 向后兼容
- 所有修改保持向后兼容
- 未指定 `prompt_type` 时默认使用 `generic`
- 现有代码无需修改即可继续运行

### Python 版本
- 需要 Python 3.7+
- 使用类型提示（`Optional`, `Dict`, `Any`）

### 依赖项
- 无新增外部依赖
- 使用标准库的类型提示

---

## 下一步工作

### 潜在改进
1. 添加更多 Prompt 模板（如特定领域的优化）
2. 支持动态 Prompt 生成（基于问题类型）
3. 添加实时性能监控
4. 实现自动化结果对比和可视化

### 实验扩展
1. 尝试不同的 top_k 参数
2. 测试不同的检索器组合
3. 评估不同模型的表现
4. 分析失败案例

---

## 文件清单

### 新增文件 (9个)
1. `benchmark_dense.sh`
2. `benchmark_sparse.sh`
3. `benchmark_hybrid.sh`
4. `benchmark_no_tool.sh`
5. `run_all_benchmarks.sh`
6. `gateway_config_rag_no_tool.json`
7. `BENCHMARK_GUIDE.md`
8. `BENCHMARK_COMPARISON.md`
9. `IMPLEMENTATION_SUMMARY.md`

### 修改文件 (3个)
1. `src/envs/http_mcp_rag_env.py`
2. `src/run_parallel_rollout.py`
3. `run_rag_benchmark.sh`

### 已有文件（未修改但需要）(3个)
1. `gateway_config_rag_dense_only.json`
2. `gateway_config_rag_sparse_only.json`
3. `gateway_config_rag_hybrid.json`

---

## 总结

本次实现成功添加了：
1. ✅ 四种 Prompt 模板以支持不同的检索策略
2. ✅ 四个独立的基准测试脚本，可单独运行
3. ✅ 完整的参数传递链（从脚本到环境）
4. ✅ 详细的文档和使用指南
5. ✅ 保持向后兼容性
6. ✅ 统一的脚本格式和配置结构

所有脚本均可独立运行，互不干扰，便于进行对比实验和性能分析。
