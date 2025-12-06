#!/bin/bash

# =================================================================
# 配置区域 (可根据测试需求修改此处)
# =================================================================
# 1. 设置测试模式: 'http_mcp_search' (搜索测试) 或 'http_mcp' (混合/VM测试)
ENV_MODE="http_mcp_search" 

# 2. 设置测试数据文件路径
DATA_PATH="search_test_demo.jsonl"

# 3. 设置网关配置文件 (注意：搜索测试用 gateway_config.json，混合测试可能需要 full/hybrid 版)
GATEWAY_CONFIG="gateway_config.json"

# 4. 其他配置
RESOURCE_PORT=8000
GATEWAY_PORT=8080
LOG_DIR="logs"
NUM_ROLLOUTS=2  # 并行 Worker 数量

# =================================================================
# 环境准备
# =================================================================
mkdir -p $LOG_DIR

# 激活 conda 环境 (请根据实际路径调整)
source /home/a1/tools/anaconda3/etc/profile.d/conda.sh
conda activate osworld_rag_lb || echo "⚠️ Conda env not found, assuming python is in path"

export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# =================================================================
# 1. 清理旧进程
# =================================================================
echo "🧹 Cleaning up ports..."
fuser -k $RESOURCE_PORT/tcp > /dev/null 2>&1
fuser -k $GATEWAY_PORT/tcp > /dev/null 2>&1
sleep 2

# =================================================================
# 2. 启动服务 (Resource API + MCP Gateway)
# =================================================================
echo "🚀 Starting Backend Services..."

# 启动 Resource API
nohup python src/services/resource_api.py > $LOG_DIR/resource_api.log 2>&1 &
echo "   - Resource API started (Port $RESOURCE_PORT)"

# 等待后端就绪
while ! nc -z localhost $RESOURCE_PORT; do sleep 1; done

# 启动 MCP Gateway
echo "🚀 Starting Gateway with config: $GATEWAY_CONFIG..."
nohup python src/mcp_server/main.py --config $GATEWAY_CONFIG --port $GATEWAY_PORT > $LOG_DIR/gateway.log 2>&1 &
echo "   - Gateway started (Port $GATEWAY_PORT)"

# 等待网关就绪
while ! nc -z localhost $GATEWAY_PORT; do sleep 1; done

# =================================================================
# 3. 执行测试 (Client Rollout)
# =================================================================
echo ""
echo "👉 Running Test: Mode=[$ENV_MODE] | Data=[$DATA_PATH]"
echo "----------------------------------------------------------------"

python src/run_parallel_rollout.py \
  --data_path $DATA_PATH \
  --num_rollouts $NUM_ROLLOUTS \
  --env_mode $ENV_MODE \
  --mcp_server_url http://localhost:$GATEWAY_PORT \
  --resource_api_url http://localhost:$RESOURCE_PORT \
  --output_dir results/test_run_$(date +%Y%m%d_%H%M%S) \
  --max_turns 10 \
  --model_name "gpt-4.1-2025-04-14" \
  2>&1 | tee $LOG_DIR/client_run.log

# =================================================================
# 4. 自动清理
# =================================================================
echo ""
echo "🛑 Cleaning up background services..."
fuser -k $RESOURCE_PORT/tcp > /dev/null 2>&1
fuser -k $GATEWAY_PORT/tcp > /dev/null 2>&1
echo "✅ Done."