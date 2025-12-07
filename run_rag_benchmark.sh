#!/bin/bash
# 启动 HTTP MCP RAG 环境测评脚本
# 使用 exact_match 和 f1_score 两种测评方案

# 设置默认参数
DATA_PATH="${DATA_PATH:-src/data/rag_demo.jsonl}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-5}"
OUTPUT_DIR="${OUTPUT_DIR:-results/rag_test_$(date +%Y%m%d_%H%M%S)}"
MODEL_NAME="${MODEL_NAME:-openai/gpt-oss-120b}"
MAX_TURNS="${MAX_TURNS:-15}"

# MCP 服务器配置
MCP_SERVER_URL="${MCP_SERVER_URL:-http://localhost:8080}"
RESOURCE_API_URL="${RESOURCE_API_URL:-http://localhost:8000}"

# 打印配置信息
echo "=================================="
echo "RAG Benchmark Configuration"
echo "=================================="
echo "Data Path: $DATA_PATH"
echo "Num Rollouts: $NUM_ROLLOUTS"
echo "Output Dir: $OUTPUT_DIR"
echo "Model Name: $MODEL_NAME"
echo "Max Turns: $MAX_TURNS"
echo "MCP Server: $MCP_SERVER_URL"
echo "Resource API: $RESOURCE_API_URL"
echo "Evaluation Metrics: exact_match, f1_score"
echo "=================================="
echo ""

# 确保输出目录存在
mkdir -p "$OUTPUT_DIR"

# 运行测评
python src/run_parallel_rollout.py \
    --data_path "$DATA_PATH" \
    --num_rollouts "$NUM_ROLLOUTS" \
    --env_mode "http_mcp_rag" \
    --output_dir "$OUTPUT_DIR" \
    --mcp_server_url "$MCP_SERVER_URL" \
    --resource_api_url "$RESOURCE_API_URL" \
    --model_name "$MODEL_NAME" \
    --max_turns "$MAX_TURNS" \
    --evaluation_metric exact_match f1_score

# 检查执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "✅ Benchmark completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    echo "=================================="
    echo ""
    echo "Files created:"
    ls -lh "$OUTPUT_DIR"
else
    echo ""
    echo "=================================="
    echo "❌ Benchmark failed!"
    echo "=================================="
    exit 1
fi
