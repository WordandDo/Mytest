#!/bin/bash
# 运行 RAG-only 环境的示例脚本

# 设置默认参数
DATA_PATH="${DATA_PATH:-src/data/HotPotQA.jsonl}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-10}"
MODEL_NAME="${MODEL_NAME:-gpt-4.1-2025-04-14}"
MAX_TURNS="${MAX_TURNS:-15}"
OUTPUT_DIR="${OUTPUT_DIR:-results/rag_only}"
MCP_SERVER_URL="${MCP_SERVER_URL:-http://localhost:8080}"

echo "=========================================="
echo "Running RAG-only Environment"
echo "=========================================="
echo "Data Path: $DATA_PATH"
echo "Num Rollouts: $NUM_ROLLOUTS"
echo "Model: $MODEL_NAME"
echo "Max Turns: $MAX_TURNS"
echo "Output Dir: $OUTPUT_DIR"
echo "MCP Server: $MCP_SERVER_URL"
echo "=========================================="

# 运行脚本
python src/run_parallel_rollout.py \
    --data_path "$DATA_PATH" \
    --num_rollouts "$NUM_ROLLOUTS" \
    --env_mode "http_mcp_rag" \
    --output_dir "$OUTPUT_DIR" \
    --mcp_server_url "$MCP_SERVER_URL" \
    --model_name "$MODEL_NAME" \
    --max_turns "$MAX_TURNS"

echo "=========================================="
echo "Execution completed. Check results in: $OUTPUT_DIR"
echo "=========================================="
