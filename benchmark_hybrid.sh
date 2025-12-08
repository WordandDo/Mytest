#!/bin/bash
# benchmark_hybrid.sh

# 1. 清理旧的 Gateway
echo "🧹 Cleaning up old gateway..."
lsof -ti:8080 | xargs kill -9 2>/dev/null

# 2. 启动 Hybrid Gateway
echo "🚀 Starting Gateway (Hybrid Mode)..."
python src/mcp_server/main.py --config gateway_config_rag_hybrid.json --port 8080 &
GATEWAY_PID=$!

# 等待启动
sleep 5
echo "✅ Gateway started with PID $GATEWAY_PID"

# 3. 运行测评
# Note: API keys are loaded from .env file by run_parallel_rollout.py
echo "📊 Running Benchmark (Hybrid)..."
export OUTPUT_DIR="results/benchmark_hybrid"
export DATA_PATH="src/data/bamboogle.json" # 或 rag_demo.jsonl
export NUM_ROLLOUTS=5
export GATEWAY_CONFIG_PATH="gateway_config_rag_hybrid.json"
export PROMPT_TYPE="hybrid"

./run_rag_benchmark.sh

# 4. 清理
echo "🛑 Stopping Gateway..."
kill $GATEWAY_PID