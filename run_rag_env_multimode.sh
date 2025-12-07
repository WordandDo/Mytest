#!/bin/bash
# RAG 环境多模式测评脚本
# 支持三种检索模式：混合(hybrid)、仅密集(dense)、仅稀疏(sparse)

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 打印使用说明
usage() {
    cat << EOF
使用方法:
  $0 [模式]

模式选项:
  hybrid    - 混合检索模式（同时使用稀疏BM25和密集E5检索）
  dense     - 仅密集检索模式（仅使用E5向量检索）
  sparse    - 仅稀疏检索模式（仅使用BM25关键词检索）
  all       - 依次运行所有三种模式（默认）

环境变量:
  DATA_PATH              - 数据集路径（默认: src/data/HotPotQA.jsonl）
  NUM_ROLLOUTS           - 并行数量（默认: 10）
  MODEL_NAME             - 模型名称（默认: gpt-4.1-2025-04-14）
  MAX_TURNS              - 最大轮次（默认: 15）
  MCP_SERVER_URL         - MCP服务器地址（默认: http://localhost:8080）
  TASK_EXECUTION_TIMEOUT - 任务超时时间（默认: 900秒）
  EVALUATION_METRICS     - 评测指标（默认: exact_match f1_score similarity contains_answer）
  BASE_OUTPUT_DIR        - 输出基础目录（默认: results/rag_multimode）

示例:
  # 运行混合模式
  $0 hybrid

  # 运行所有模式
  $0 all

  # 使用自定义参数运行密集模式
  NUM_ROLLOUTS=20 MODEL_NAME=gpt-4 $0 dense

EOF
    exit 1
}

# 设置默认参数
DATA_PATH="${DATA_PATH:-src/data/bamboogle.json}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-10}"
MODEL_NAME="${MODEL_NAME:-gpt-4.1-2025-04-14}"
MAX_TURNS="${MAX_TURNS:-15}"
MCP_SERVER_URL="${MCP_SERVER_URL:-http://localhost:8080}"
TASK_EXECUTION_TIMEOUT="${TASK_EXECUTION_TIMEOUT:-900}"
EVALUATION_METRICS="${EVALUATION_METRICS:-exact_match f1_score}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-results/rag_multimode}"

# 时间戳（用于区分不同批次的运行）
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 检查数据文件是否存在
if [ ! -f "$DATA_PATH" ]; then
    print_error "数据文件不存在: $DATA_PATH"
    exit 1
fi

# 运行单个模式的函数
run_mode() {
    local MODE=$1
    local GATEWAY_CONFIG=""
    local OUTPUT_DIR="${BASE_OUTPUT_DIR}/${MODE}_${TIMESTAMP}"

    # 根据模式选择对应的 gateway 配置
    case $MODE in
        hybrid)
            GATEWAY_CONFIG="gateway_config_rag_hybrid.json"
            print_info "运行模式: 混合检索 (稀疏BM25 + 密集E5)"
            ;;
        dense)
            GATEWAY_CONFIG="gateway_config_rag_dense_only.json"
            print_info "运行模式: 仅密集检索 (E5向量检索)"
            ;;
        sparse)
            GATEWAY_CONFIG="gateway_config_rag_sparse_only.json"
            print_info "运行模式: 仅稀疏检索 (BM25关键词检索)"
            ;;
        *)
            print_error "未知模式: $MODE"
            return 1
            ;;
    esac

    # 检查 gateway 配置文件是否存在
    if [ ! -f "$GATEWAY_CONFIG" ]; then
        print_error "Gateway 配置文件不存在: $GATEWAY_CONFIG"
        return 1
    fi

    print_info "=========================================="
    print_info "RAG Environment - ${MODE^^} Mode"
    print_info "=========================================="
    print_info "Gateway Config: $GATEWAY_CONFIG"
    print_info "Data Path: $DATA_PATH"
    print_info "Num Rollouts: $NUM_ROLLOUTS"
    print_info "Model: $MODEL_NAME"
    print_info "Max Turns: $MAX_TURNS"
    print_info "Task Timeout: $TASK_EXECUTION_TIMEOUT seconds"
    print_info "Output Dir: $OUTPUT_DIR"
    print_info "MCP Server: $MCP_SERVER_URL"
    print_info "Evaluation Metrics: $EVALUATION_METRICS"
    print_info "=========================================="

    # 创建输出目录
    mkdir -p "$OUTPUT_DIR"

    # 保存配置信息到文件
    cat > "${OUTPUT_DIR}/config.txt" << EOF
运行时间: $(date '+%Y-%m-%d %H:%M:%S')
运行模式: ${MODE}
Gateway配置: ${GATEWAY_CONFIG}
数据路径: ${DATA_PATH}
并行数量: ${NUM_ROLLOUTS}
模型名称: ${MODEL_NAME}
最大轮次: ${MAX_TURNS}
任务超时: ${TASK_EXECUTION_TIMEOUT}秒
MCP服务器: ${MCP_SERVER_URL}
评测指标: ${EVALUATION_METRICS}
EOF

    # 运行测评
    print_info "开始执行..."

    python src/run_parallel_rollout.py \
        --data_path "$DATA_PATH" \
        --num_rollouts "$NUM_ROLLOUTS" \
        --env_mode "http_mcp_rag" \
        --output_dir "$OUTPUT_DIR" \
        --mcp_server_url "$MCP_SERVER_URL" \
        --gateway_config_path "$GATEWAY_CONFIG" \
        --model_name "$MODEL_NAME" \
        --max_turns "$MAX_TURNS" \
        --evaluation_metric $EVALUATION_METRICS

    local EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        print_success "模式 ${MODE} 执行完成"
        print_info "结果保存在: $OUTPUT_DIR"
        print_info "  - evaluation_scores.json: 每个任务的详细评分"
        print_info "  - evaluation_summary.json: 所有指标的汇总统计"
        print_info "  - trajectory.jsonl: 执行轨迹"
        print_info "  - config.txt: 运行配置信息"
    else
        print_error "模式 ${MODE} 执行失败 (退出码: $EXIT_CODE)"
        return $EXIT_CODE
    fi

    echo ""
    return 0
}

# 主逻辑
MODE="${1:-all}"

case $MODE in
    hybrid|dense|sparse)
        run_mode "$MODE"
        EXIT_CODE=$?
        ;;
    all)
        print_info "=========================================="
        print_info "运行所有三种模式"
        print_info "批次时间戳: $TIMESTAMP"
        print_info "=========================================="
        echo ""

        FAILED_MODES=""

        # 依次运行三种模式
        for mode in hybrid dense sparse; do
            run_mode "$mode"
            if [ $? -ne 0 ]; then
                FAILED_MODES="$FAILED_MODES $mode"
            fi
            echo ""
        done

        # 汇总结果
        print_info "=========================================="
        print_info "所有模式执行完成"
        print_info "=========================================="
        print_info "结果目录: $BASE_OUTPUT_DIR"

        if [ -z "$FAILED_MODES" ]; then
            print_success "所有模式均成功完成！"
            ls -lh "$BASE_OUTPUT_DIR"
            EXIT_CODE=0
        else
            print_warning "以下模式执行失败:$FAILED_MODES"
            EXIT_CODE=1
        fi

        # 生成汇总报告
        SUMMARY_FILE="${BASE_OUTPUT_DIR}/summary_${TIMESTAMP}.txt"
        {
            echo "RAG 多模式测评汇总报告"
            echo "======================================"
            echo "运行时间: $(date '+%Y-%m-%d %H:%M:%S')"
            echo "批次时间戳: $TIMESTAMP"
            echo ""
            echo "测评配置:"
            echo "  数据集: $DATA_PATH"
            echo "  模型: $MODEL_NAME"
            echo "  并行数: $NUM_ROLLOUTS"
            echo "  评测指标: $EVALUATION_METRICS"
            echo ""
            echo "各模式结果目录:"
            for mode in hybrid dense sparse; do
                mode_dir="${BASE_OUTPUT_DIR}/${mode}_${TIMESTAMP}"
                if [ -d "$mode_dir" ]; then
                    echo "  [$mode] $mode_dir"
                fi
            done
            echo ""
            if [ -z "$FAILED_MODES" ]; then
                echo "状态: 全部成功 ✓"
            else
                echo "状态: 部分失败"
                echo "失败模式:$FAILED_MODES"
            fi
        } > "$SUMMARY_FILE"

        print_info "汇总报告已保存: $SUMMARY_FILE"
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        print_error "未知模式: $MODE"
        echo ""
        usage
        ;;
esac

exit $EXIT_CODE
