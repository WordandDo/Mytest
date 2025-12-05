#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# 1. 检查后端是否就绪
echo "🔍 Checking backend resource status..."
python src/utils/wait_for_backend.py

# 检查上一条命令的退出代码
if [ $? -ne 0 ]; then
    echo "❌ Backend failed to initialize within timeout. Gateway startup aborted."
    exit 1
fi

# 2. 后端就绪后，启动 Gateway
echo "🚀 Backend is ready. Starting MCP Gateway..."
# 您的原始启动命令
python src/mcp_server/main.py --config gateway_config.json --port 8080