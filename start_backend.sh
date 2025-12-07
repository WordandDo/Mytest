#!/bin/bash
# 设置 Python 路径
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# 配置参数
RESOURCE_PORT=8000
RAG_SERVICE_PORT=8001
MAX_WAIT_TIME=600  # 最大等待时间（秒）
HEALTH_CHECK_INTERVAL=2  # 健康检查间隔（秒）

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

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

# 启动 Resource API 服务
print_info "🚀 Starting Resource API on port ${RESOURCE_PORT}..."
python src/services/resource_api.py &
API_PID=$!

# 等待 Resource API 服务启动
print_info "Waiting for Resource API to start..."
elapsed=0
while [ $elapsed -lt $MAX_WAIT_TIME ]; do
    if nc -z localhost $RESOURCE_PORT 2>/dev/null; then
        print_success "Resource API is listening on port ${RESOURCE_PORT}"
        break
    fi
    sleep 1
    elapsed=$((elapsed + 1))

    # 每10秒显示一次进度
    if [ $((elapsed % 10)) -eq 0 ]; then
        print_info "Still waiting... (${elapsed}s elapsed)"
    fi
done

if [ $elapsed -ge $MAX_WAIT_TIME ]; then
    print_error "Timeout waiting for Resource API to start"
    kill $API_PID 2>/dev/null
    exit 1
fi

# 等待 RAG 服务完全就绪（索引加载完成）
print_info "Waiting for RAG service to be fully ready (index loading)..."
elapsed=0
rag_ready=false

while [ $elapsed -lt $MAX_WAIT_TIME ]; do
    # 检查 RAG 服务端口是否监听
    if nc -z localhost $RAG_SERVICE_PORT 2>/dev/null; then
        # 端口已监听，检查健康状态
        health_response=$(curl -s http://localhost:${RAG_SERVICE_PORT}/health 2>/dev/null)

        if [ $? -eq 0 ]; then
            # 检查 ready 字段是否为 true
            ready_status=$(echo "$health_response" | grep -o '"ready":\s*true')

            if [ -n "$ready_status" ]; then
                print_success "RAG service is fully ready (index loaded)"
                rag_ready=true
                break
            else
                print_warning "RAG service started but index is still loading..."
            fi
        fi
    fi

    sleep $HEALTH_CHECK_INTERVAL
    elapsed=$((elapsed + HEALTH_CHECK_INTERVAL))

    # 每20秒显示一次进度
    if [ $((elapsed % 20)) -eq 0 ]; then
        print_info "Still waiting for RAG index to load... (${elapsed}s elapsed)"
    fi
done

if [ "$rag_ready" = false ]; then
    print_warning "RAG service did not become ready within ${MAX_WAIT_TIME}s"
    print_warning "Service may still be loading. Check logs for details."
fi

# 执行资源预热测试
print_info "Performing resource warmup test..."
python -c "
import requests
import sys

try:
    # 测试 RAG 查询
    response = requests.post(
        'http://localhost:${RAG_SERVICE_PORT}/query',
        json={'query': 'test warmup query', 'top_k': 1, 'search_type': 'dense'},
        timeout=30
    )

    if response.status_code == 200:
        print('✅ RAG warmup query successful')
        sys.exit(0)
    else:
        print(f'⚠️  RAG warmup query returned status {response.status_code}')
        sys.exit(1)
except Exception as e:
    print(f'❌ RAG warmup query failed: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    print_success "Resource warmup completed successfully"
else
    print_warning "Resource warmup had issues, but services are running"
fi

# 显示服务状态
echo ""
print_success "=========================================="
print_success "Backend Services Ready"
print_success "=========================================="
print_info "Resource API:  http://localhost:${RESOURCE_PORT}"
print_info "RAG Service:   http://localhost:${RAG_SERVICE_PORT}"
print_info "Resource API PID: ${API_PID}"
echo ""
print_info "To stop services: kill ${API_PID}"
print_info "Press Ctrl+C to stop..."
echo ""

# 保持脚本运行，等待用户中断
wait $API_PID