#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# 确保 Resource API 已经运行
# 启动 Gateway Server (读取 gateway_config.json)
echo "🚀 Starting Composite Gateway Server..."
python src/mcp_server/main.py --config gateway_config.json --port 8080