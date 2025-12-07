#!/bin/bash
# benchmark_sparse.sh

# 1. 清理旧的 Gateway
echo "🧹 Cleaning up old gateway..."
lsof -ti:8080 | xargs kill -9 2>/dev/null

# 2. 启动 Sparse-Only Gateway
echo "🚀 Starting Gateway (Sparse Only)..."
python src/mcp_server/main.py --config gateway_config_rag_sparse_only.json --port 8080 &
GATEWAY_PID=$!

# 等待启动
sleep 5
echo "✅ Gateway started with PID $GATEWAY_PID"

# 3. 运行测评
echo "📊 Running Benchmark (Sparse)..."
export OUTPUT_DIR="results/benchmark_sparse_only"
export DATA_PATH="src/data/bamboogle.json" # 或 rag_demo.jsonl
export NUM_ROLLOUTS=5

./run_rag_benchmark.sh

# 4. 清理
echo "🛑 Stopping Gateway..."
kill $GATEWAY_PID