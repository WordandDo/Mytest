# RAG 环境测评配置总结

## 📝 修改内容

本次配置为 `src/envs/http_mcp_rag_env.py` 环境创建了完整的测评脚本和文档，支持 `exact_match` 和 `f1_score` 两种测评方案。

### 1. 创建的脚本文件

#### [run_rag_benchmark.sh](run_rag_benchmark.sh)
- **类型**: Bash 启动脚本
- **功能**:
  - 使用环境变量配置参数
  - 调用 `src/run_parallel_rollout.py` 运行测评
  - 支持 `exact_match` 和 `f1_score` 双指标评测
  - 自动创建时间戳输出目录
- **使用方法**:
  ```bash
  ./run_rag_benchmark.sh
  # 或自定义配置
  DATA_PATH=src/data/HotPotQA.jsonl NUM_ROLLOUTS=10 ./run_rag_benchmark.sh
  ```

#### [run_rag_test.py](run_rag_test.py)
- **类型**: Python 启动脚本
- **功能**:
  - 提供 Python 接口
  - 支持环境变量配置
  - 详细的日志输出
  - 返回测评结果统计
- **使用方法**:
  ```bash
  python3 run_rag_test.py
  # 或通过环境变量
  DATA_PATH=src/data/bamboogle.json python3 run_rag_test.py
  ```

#### [demo_rag_test.sh](demo_rag_test.sh)
- **类型**: 交互式演示脚本
- **功能**:
  - 显示数据集样本
  - 检查 RAG 配置状态
  - 交互式确认执行
  - 适合新手使用
- **使用方法**:
  ```bash
  ./demo_rag_test.sh
  ```

### 2. 创建的文档文件

#### [RAG_BENCHMARK_GUIDE.md](RAG_BENCHMARK_GUIDE.md)
- **类型**: 详细使用指南
- **内容**:
  - 完整的配置说明
  - 所有参数详解
  - 测评指标说明
  - 输出文件格式
  - 常见问题解答
  - 性能优化建议

#### [QUICKSTART.md](QUICKSTART.md)
- **类型**: 快速开始指南
- **内容**:
  - 快速运行方法
  - 必需配置项
  - 常用示例
  - 推荐工作流

### 3. 配置修改

#### [deployment_config.json](deployment_config.json:47-62)
- **修改**: 启用 RAG 资源
- **改动**:
  ```json
  "rag": {
    "enabled": true,  // 由 false 改为 true
    ...
  }
  ```

## 🎯 核心特性

### 双指标测评
配置支持同时使用两种测评指标：

1. **Exact Match (精确匹配)**
   - 实现位置: [src/benchmark/benchmark.py:387-389](src/benchmark/benchmark.py:387-389)
   - 标准化处理: [src/benchmark/benchmark.py:23-44](src/benchmark/benchmark.py:23-44)
   - 得分类型: 二值 (0 或 1)

2. **F1 Score (F1 分数)**
   - 实现位置: [src/benchmark/benchmark.py:391-394](src/benchmark/benchmark.py:391-394)
   - 计算逻辑: [src/benchmark/benchmark.py:63-95](src/benchmark/benchmark.py:63-95)
   - 得分类型: 连续值 (0.0-1.0)

### 并行执行框架
- 基于: [src/run_parallel_rollout.py](src/run_parallel_rollout.py)
- 支持多 worker 并行
- 自动资源分配与释放
- 完整的错误处理

### RAG 环境集成
- 环境类: [src/envs/http_mcp_rag_env.py](src/envs/http_mcp_rag_env.py)
- 基于 HTTP MCP 协议
- 自动过滤只使用 RAG 资源
- 专用系统提示词

## 📊 输出结果说明

### 生成的文件

所有输出保存在 `results/<测试名称>/` 目录：

```
results/rag_test_20251207_103000/
├── trajectory.jsonl              # 完整执行轨迹
├── evaluation_scores.json        # 详细评分（每个任务）
├── evaluation_summary.json       # 汇总统计（所有指标）
└── worker_instance_map.json      # Worker 资源映射
```

### evaluation_scores.json 格式

```json
[
  {
    "task_id": "5a8b57f25542995d1e6f1371",
    "predicted_answer": "yes",
    "ground_truth": "yes",
    "scores": {
      "exact_match": 1.0,
      "f1_score": 1.0
    },
    "is_correct": {
      "exact_match": true,
      "f1_score": true
    }
  }
]
```

### evaluation_summary.json 格式

```json
{
  "timestamp": "2025-12-07T10:30:00.123456",
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

## 🔧 配置参数

### 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `DATA_PATH` | `src/data/rag_demo.jsonl` | 测试数据路径 |
| `NUM_ROLLOUTS` | `5` | 并行 worker 数量 |
| `OUTPUT_DIR` | `results/rag_test_<时间戳>` | 输出目录 |
| `MODEL_NAME` | `gpt-4.1-2025-04-14` | LLM 模型名称 |
| `MAX_TURNS` | `15` | 每任务最大轮次 |
| `MCP_SERVER_URL` | `http://localhost:8080` | MCP 服务器地址 |
| `RESOURCE_API_URL` | `http://localhost:8000` | 资源 API 地址 |

### 命令行参数

```bash
--data_path          # 测试数据路径
--num_rollouts       # 并行 worker 数量
--env_mode           # 环境模式（http_mcp_rag）
--output_dir         # 输出目录
--model_name         # 模型名称
--max_turns          # 最大轮次
--evaluation_metric  # 测评指标（可多个）
--mcp_server_url     # MCP 服务器 URL
--resource_api_url   # 资源 API URL
```

## 📚 可用数据集

| 文件 | 描述 | 数量 | 用途 |
|------|------|------|------|
| `src/data/rag_demo.jsonl` | RAG 演示数据 | 5 | 快速测试 |
| `src/data/HotPotQA_demo.jsonl` | HotPotQA 样本 | ~10 | 功能测试 |
| `src/data/HotPotQA.jsonl` | HotPotQA 完整 | 数千 | 完整评测 |
| `src/data/bamboogle.json` | Bamboogle 数据 | 数百 | 专项测试 |

## 🚀 使用流程

### 快速开始（推荐）

1. **启动资源服务器**
   ```bash
   python3 -m utils.resource_pools.gateway_server
   ```

2. **运行演示脚本**
   ```bash
   ./demo_rag_test.sh
   ```

3. **查看结果**
   ```bash
   cat results/demo_*/evaluation_summary.json | python3 -m json.tool
   ```

### 完整测评流程

1. **准备环境**
   ```bash
   # 配置 .env 文件
   echo "OPENAI_API_KEY=your_key" > .env

   # 启动资源服务器
   python3 -m utils.resource_pools.gateway_server &
   ```

2. **运行测评**
   ```bash
   DATA_PATH=src/data/HotPotQA.jsonl \
   NUM_ROLLOUTS=20 \
   OUTPUT_DIR=results/production_test \
   ./run_rag_benchmark.sh
   ```

3. **分析结果**
   ```bash
   # 查看汇总
   cat results/production_test/evaluation_summary.json | python3 -m json.tool

   # 查看详细评分
   cat results/production_test/evaluation_scores.json | python3 -m json.tool

   # 统计成功率
   jq '.[] | select(.is_correct.exact_match == true)' results/production_test/evaluation_scores.json | wc -l
   ```

## ✅ 验证清单

运行前检查：

- [ ] RAG 资源已在 `deployment_config.json` 中启用
- [ ] 资源服务器已启动
- [ ] `.env` 文件已配置 API 密钥
- [ ] 测试数据文件存在
- [ ] RAG 索引路径正确

## 🐛 故障排查

### 问题 1: RAG 资源未启用
```bash
# 检查配置
grep -A 2 '"rag"' deployment_config.json | grep enabled

# 应该显示: "enabled": true
```

### 问题 2: 资源服务器未启动
```bash
# 检查进程
ps aux | grep gateway_server

# 启动服务器
python3 -m utils.resource_pools.gateway_server
```

### 问题 3: 数据文件不存在
```bash
# 列出可用数据集
ls -lh src/data/*.json*
```

### 问题 4: API 密钥未配置
```bash
# 检查环境变量
grep OPENAI_API_KEY .env
```

## 📈 性能优化

1. **调整并行度**
   - CPU 核心数较多: `NUM_ROLLOUTS=20`
   - 内存受限: `NUM_ROLLOUTS=5`

2. **使用 GPU 加速**
   - 在 `deployment_config.json` 中设置 `use_gpu_index: true`

3. **减少检索文档数**
   - 修改 `default_top_k` 参数

4. **批量处理大数据集**
   - 使用更高的 `NUM_ROLLOUTS` 值

## 📞 技术支持

如遇问题，请检查：
1. 日志输出中的错误信息
2. `deployment_config.json` 配置
3. 资源服务器状态
4. 环境变量设置

---

## 📌 快速参考

### 最简单的运行方式
```bash
./run_rag_benchmark.sh
```

### 最常用的运行方式
```bash
DATA_PATH=src/data/HotPotQA.jsonl \
NUM_ROLLOUTS=10 \
./run_rag_benchmark.sh
```

### 查看结果
```bash
cat results/*/evaluation_summary.json | python3 -m json.tool
```

---

**创建时间**: 2025-12-07
**环境版本**: http_mcp_rag
**测评指标**: exact_match, f1_score
**状态**: ✅ 已完成并测试
