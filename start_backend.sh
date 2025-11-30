#!/bin/bash
# 设置 Python 路径
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# 启动 Resource API 服务
# 默认端口为 8000，rag_server.py 会连接这个端口
echo "🚀 Starting Resource API..."
python src/services/resource_api.py