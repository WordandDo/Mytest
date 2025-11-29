#!/bin/bash

# =================================================================
# 配置区域 (可根据需要修改)
# =================================================================

# 1. 资源池配置
export NUM_VMS=1                    # 启动 10 台虚拟机 (M值)
export PROVIDER_NAME="aliyun"         # 云厂商: aliyun, aws, 或 docker
export rag_pool_size=5              # RAG 资源池大小
export rag_worker_size=5            # RAG Worker 数量
# 2. 服务地址配置
export RESOURCE_API_HOST="0.0.0.0"
export RESOURCE_API_PORT=8000
export RESOURCE_API_URL="http://localhost:${RESOURCE_API_PORT}"

export MCP_SERVER_HOST="0.0.0.0"
export MCP_SERVER_PORT=8080
export MCP_SERVER_URL="http://localhost:${MCP_SERVER_PORT}"

# 3. 任务配置
DATA_PATH="/home/lb/AgentFlow/src/data/osworld_examples.jsonl"
NUM_ROLLOUTS=2                       # 并发 Worker 数量 (N值)
OUTPUT_DIR="results/parallel_run_$(date +%Y%m%d_%H%M%S)"

# =================================================================
# 辅助函数
# =================================================================

# 定义清理函数：脚本退出或被中断时，杀死后台进程
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    
    if [ -n "$PID_MCP" ]; then
        echo "   - Killing MCP Server (PID $PID_MCP)..."
        kill $PID_MCP 2>/dev/null
    fi
    
    if [ -n "$PID_RES" ]; then
        echo "   - Killing Resource API (PID $PID_RES)..."
        kill $PID_RES 2>/dev/null
    fi
    
    echo "✅ All services stopped."
    exit
}

# 注册信号捕获 (Ctrl+C, Kill 等)
trap cleanup SIGINT SIGTERM EXIT

# 等待端口就绪的函数
wait_for_port() {
    local port=$1
    local name=$2
    local timeout=300
    local count=0
    
    echo -n "⏳ Waiting for $name to start on port $port..."
    while ! nc -z localhost $port; do
        sleep 1
        count=$((count+1))
        if [ $count -ge $timeout ]; then
            echo " Timeout!"
            echo "❌ Error: $name failed to start."
            exit 1
        fi
        echo -n "."
    done
    echo " Ready!"
}

# =================================================================
# 启动流程
# =================================================================

echo "==========================================================="
echo "🚀 Starting OSWorld Parallel System"
echo "   - VMs (M): $NUM_VMS"
echo "   - Workers (N): $NUM_ROLLOUTS"
echo "==========================================================="

# 1. 启动 Resource API (资源管理层)
echo "[1/3] Starting Resource API..."
python src/services/resource_api.py > resource_api.log 2>&1 &
PID_RES=$!
echo "   - PID: $PID_RES"
echo "   - Log: resource_api.log"

# 等待 Resource API 就绪 (这是必要的，因为 MCP Server 启动时可能不依赖它，但 Worker 需要)
wait_for_port $RESOURCE_API_PORT "Resource API"

# 2. 启动 MCP Server (网关层)
echo "[2/3] Starting MCP Server Gateway..."
# 注意：这里运行的是修改后支持 Uvicorn 的 server 文件
python src/mcp_server/osworld_server.py > mcp_server.log 2>&1 &
PID_MCP=$!
echo "   - PID: $PID_MCP"
echo "   - Log: mcp_server.log"

# 等待 MCP Server 就绪
wait_for_port $MCP_SERVER_PORT "MCP Server"

# 3. 启动并行 Rollout (执行层)
echo "[3/3] Launching Parallel Workers..."
echo "   - Data: $DATA_PATH"
echo "   - Output: $OUTPUT_DIR"
echo "-----------------------------------------------------------"

# 确保输出目录存在
mkdir -p "$OUTPUT_DIR"

# 运行主脚本 (前台运行)
python src/run_parallel_rollout.py \
  --data_path "$DATA_PATH" \
  --num_rollouts "$NUM_ROLLOUTS" \
  --env_mode http_mcp \
  --mcp_server_url "$MCP_SERVER_URL" \
  --resource_api_url "$RESOURCE_API_URL" \
  --output_dir "$OUTPUT_DIR"

# =================================================================
# 结束
# =================================================================
# 当 run_parallel_rollout.py 运行结束时，脚本会继续执行到这里
# 此时 trap EXIT 会被触发，自动调用 cleanup 函数清理后台服务
echo "🎉 All tasks completed successfully!"