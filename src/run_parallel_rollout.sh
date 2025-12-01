#!/bin/bash

# =================================================================
# 配置区域
# =================================================================
RESOURCE_PORT=8000
GATEWAY_PORT=8080
LOG_DIR="logs"

# 确保日志目录存在
mkdir -p $LOG_DIR

# 设置 Python 路径
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# =================================================================
# 1. 环境清理 (杀掉旧进程)
# =================================================================
echo "🧹 [1/3] Cleaning up ports $RESOURCE_PORT and $GATEWAY_PORT..."

# 使用 fuser 杀掉占用端口的进程 (如果是 MacOS 请改用 lsof -i :port | awk 'NR!=1 {print $2}' | xargs kill)
fuser -k $RESOURCE_PORT/tcp > /dev/null 2>&1
fuser -k $GATEWAY_PORT/tcp > /dev/null 2>&1

# 等待进程完全释放
sleep 2
echo "   - Ports cleared."

# =================================================================
# 2. 启动 Resource API (后端)
# =================================================================
echo "🚀 [2/3] Starting Resource API on port $RESOURCE_PORT..."

# 后台启动并重定向日志
nohup python src/services/resource_api.py > $LOG_DIR/resource_api.log 2>&1 &
PID_RES=$!
echo "   - Resource API PID: $PID_RES"

# 循环检查端口是否就绪
echo -n "   - Waiting for service readiness..."
count=0
while ! nc -z localhost $RESOURCE_PORT; do   
  sleep 1
  echo -n "."
  count=$((count+1))
  if [ $count -ge 300 ]; then
      echo " ❌ Timeout! Resource API failed to start. Check $LOG_DIR/resource_api.log"
      exit 1
  fi
done
echo " ✅ Ready!"

# =================================================================
# 3. 启动 MCP Gateway (网关)
# =================================================================
echo "🚀 [3/3] Starting MCP Gateway on port $GATEWAY_PORT..."

# 使用 gateway_config.json 启动复合网关 (同时支持 RAG 和 VM)
nohup python src/mcp_server/main.py --config gateway_config.json --port $GATEWAY_PORT > $LOG_DIR/gateway.log 2>&1 &
PID_GW=$!
echo "   - Gateway PID: $PID_GW"

# 循环检查端口是否就绪
echo -n "   - Waiting for service readiness..."
count=0
while ! nc -z localhost $GATEWAY_PORT; do   
  sleep 1
  echo -n "."
  count=$((count+1))
  if [ $count -ge 30 ]; then
      echo " ❌ Timeout! Gateway failed to start. Check $LOG_DIR/gateway.log"
      kill $PID_RES # 启动失败时清理后端
      exit 1
  fi
done
echo " ✅ Ready!"

# =================================================================
# 4. 准备就绪，打印运行指令
# =================================================================
echo ""
echo "🎉 Server Environment Established Successfully!"
echo "   - Resource API: http://localhost:$RESOURCE_PORT"
echo "   - MCP Gateway:  http://localhost:$GATEWAY_PORT/sse"
echo "   - Logs:         $LOG_DIR/"
echo ""
echo "👉 Now running your rollout script:"
echo "----------------------------------------------------------------"

# 编辑此部分
python src/run_parallel_rollout.py \
  --data_path hybrid_test_demo.jsonl \
  --num_rollouts 3 \
  --env_mode http_mcp \
  --mcp_server_url http://localhost:8080 \
  --resource_api_url http://localhost:8000 \
  --output_dir results/test_run_hybrid
# (可选) 脚本运行完后自动清理后台服务
# Uncomment lines below if you want auto-cleanup
# echo ""
# echo "🛑 Cleaning up services..."
# kill $PID_GW $PID_RES
# echo "✅ Services stopped."