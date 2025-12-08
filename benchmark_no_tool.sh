#!/bin/bash
# benchmark_no_tool.sh

# 1. 清理旧的 Gateway
echo "🧹 Cleaning up old gateway..."
lsof -ti:8080 | xargs kill -9 2>/dev/null

# 2. 启动 No-Tool Gateway (只包含系统工具)
echo "🚀 Starting Gateway (No-Tool Mode)..."
python src/mcp_server/main.py --config gateway_config_rag_no_tool.json --port 8080 &
GATEWAY_PID=$!

# 等待启动
sleep 5
echo "✅ Gateway started with PID $GATEWAY_PID"

# 3. 运行测评
echo "📊 Running Benchmark (No Tool - Pure LLM)..."
export OUTPUT_DIR="results/benchmark_no_tool"
export DATA_PATH="src/data/bamboogle.json" # 或 rag_demo.jsonl
export NUM_ROLLOUTS=10
export GATEWAY_CONFIG_PATH="gateway_config_rag_no_tool.json"
export PROMPT_TYPE="no_tool"

./run_rag_benchmark.sh

# 4. 清理
echo "🛑 Stopping Gateway..."
kill $GATEWAY_PID

echo "✅ Benchmark completed (No Tool mode)"
