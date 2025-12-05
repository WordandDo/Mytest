#!/bin/bash
# 运行 RAG-only 环境的示例脚本

# 设置默认参数
DATA_PATH="${DATA_PATH:-src/data/HotPotQA.jsonl}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-50}"
MODEL_NAME="${MODEL_NAME:-gpt-4.1-2025-04-14}"
MAX_TURNS="${MAX_TURNS:-15}"
OUTPUT_DIR="${OUTPUT_DIR:-results/rag_only}"
MCP_SERVER_URL="${MCP_SERVER_URL:-http://localhost:8080}"

# 评测指标：支持多个指标，用空格分隔
# 可选指标：exact_match, f1_score, bleu_score, rouge_score, similarity, contains_answer, numeric_match, llm_judgement
EVALUATION_METRICS="${EVALUATION_METRICS:-exact_match f1_score similarity contains_answer}"

echo "=========================================="
echo "Running RAG-only Environment"
echo "=========================================="
echo "Data Path: $DATA_PATH"
echo "Num Rollouts: $NUM_ROLLOUTS"
echo "Model: $MODEL_NAME"
echo "Max Turns: $MAX_TURNS"
echo "Output Dir: $OUTPUT_DIR"
echo "MCP Server: $MCP_SERVER_URL"
echo "Evaluation Metrics: $EVALUATION_METRICS"
echo "=========================================="

# 运行脚本
python src/run_parallel_rollout.py \
    --data_path "$DATA_PATH" \
    --num_rollouts "$NUM_ROLLOUTS" \
    --env_mode "http_mcp_rag" \
    --output_dir "$OUTPUT_DIR" \
    --mcp_server_url "$MCP_SERVER_URL" \
    --model_name "$MODEL_NAME" \
    --max_turns "$MAX_TURNS" \
    --evaluation_metric $EVALUATION_METRICS

echo "=========================================="
echo "Execution completed. Check results in: $OUTPUT_DIR"
echo "  - evaluation_scores.json: 每个任务的详细评分"
echo "  - evaluation_summary.json: 所有指标的汇总统计"
echo "  - trajectory.jsonl: 执行轨迹"
echo "=========================================="
