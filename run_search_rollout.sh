#!/bin/bash

# ==========================================
# HttpMCPSearchEnv 批量测试脚本
# ==========================================

# 0. Determine Python executable
# Priority: 1. Current shell's python, 2. python3, 3. python
if command -v python &> /dev/null && python -c "import openai" &> /dev/null; then
    PYTHON_CMD="python"
elif command -v python3 &> /dev/null && python3 -c "import openai" &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "❌ Error: Could not find Python with required packages (openai, etc.)"
    echo "   Please activate your conda environment first:"
    echo "   conda activate osworld_rag_lb"
    echo "   Then run this script again."
    exit 1
fi

echo "🐍 Using Python: $PYTHON_CMD ($(which $PYTHON_CMD))"

# 1. 基础环境设置
export PYTHONPATH=$(pwd)

# 确保 API Key 存在 (根据需要取消注释并填入，或依赖 .env 文件)
# export SERPAPI_API_KEY="your_serpapi_key"
# export OPENAI_API_KEY="your_openai_key"

# 2. 检查配置文件
if [ ! -f "gateway_config.json" ]; then
    echo "❌ Error: gateway_config.json not found!"
    echo "   Please ensure you are in the project root and the config exists."
    exit 1
fi

# 3. 定义测试参数
# ------------------------------------------
# 环境模式：对应我们在 factory.py 中注册的名称
ENV_MODE="http_mcp_search" 
# 测试数据：上面创建的 jsonl 文件
TEST_FILE="src/data/search_test_cases.jsonl"
# 模型：建议使用擅长工具调用的模型
MODEL_NAME="gpt-4o" 
# 并发度：搜索任务通常响应较快，可以适当提高，但为了调试建议先设为 1
PARALLEL_DEGREE=1
# 输出目录：自动带上时间戳
OUTPUT_DIR="results/search_rollout_$(date +%Y%m%d_%H%M%S)"

# 4. 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "🚀 Starting Search Environment Rollout..."
echo "========================================"
echo "🌍 Env Mode:      $ENV_MODE"
echo "📂 Test File:     $TEST_FILE"
echo "🤖 Model:         $MODEL_NAME"
echo "⚡ Parallelism:   $PARALLEL_DEGREE"
echo "📂 Output Dir:    $OUTPUT_DIR"
echo "========================================"

# 5. 执行 Rollout
$PYTHON_CMD src/run_parallel_rollout.py \
    --env-mode "$ENV_MODE" \
    --model-name "$MODEL_NAME" \
    --test-file "$TEST_FILE" \
    --max-turns 10 \
    --max-retries 3 \
    --parallel-degree "$PARALLEL_DEGREE" \
    --output-dir "$OUTPUT_DIR" \
    --gateway-config "gateway_config.json"

# 6. 结果提示
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Rollout completed successfully."
    echo "   Check detailed logs in: $OUTPUT_DIR"
    echo "   Use 'view_tool_stats.py' (if available) to analyze tool usage."
else
    echo ""
    echo "❌ Rollout failed with error code $?."
fi