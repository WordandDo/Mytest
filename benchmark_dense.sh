#!/bin/bash
# benchmark_dense.sh

# 1. 清理旧的 Gateway (端口 8080)
echo "🧹 Cleaning up old gateway..."
lsof -ti:8080 | xargs kill -9 2>/dev/null

# 2. 启动 Dense-Only Gateway
echo "🚀 Starting Gateway (Dense Only)..."
python src/mcp_server/main.py --config gateway_config_rag_dense_only.json --port 8080 &
GATEWAY_PID=$!

# 等待启动
sleep 5
echo "✅ Gateway started with PID $GATEWAY_PID"

# 3. 运行测评
echo "📊 Running Benchmark (Dense)..."
# 配置参数
export OUTPUT_DIR="results/benchmark_dense_only"
export DATA_PATH="src/data/bamboogle.json" # 或 rag_demo.jsonl
export NUM_ROLLOUTS=5

# 调用现有的测评脚本
./run_rag_benchmark.sh

# 4. 清理
echo "🛑 Stopping Gateway..."
kill $GATEWAY_PID