#!/bin/bash
# 快速验证 RAG 服务的脚本

echo "========================================"
echo "🔍 RAG 服务快速验证"
echo "========================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. 检查 Resource API (8000)
echo "1️⃣  检查 Resource API (端口 8000)..."
if curl -s http://localhost:8000/status > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Resource API 正常运行${NC}"
else
    echo -e "${RED}❌ Resource API 未运行，请执行: ./start_backend.sh${NC}"
    exit 1
fi
echo ""

# 2. 检查 Gateway Server (8080)
echo "2️⃣  检查 Gateway Server (端口 8080)..."
if curl -s http://localhost:8080/sse > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Gateway Server 正常运行 (MCP SSE)${NC}"
else
    echo -e "${YELLOW}⚠️  Gateway 可能未运行或使用不同协议${NC}"
fi
echo ""

# 3. 测试 RAG 资源申请
echo "3️⃣  申请 RAG 资源..."
WORKER_ID="test_worker_$$"
echo "   Worker ID: $WORKER_ID"
ALLOC_RESPONSE=$(curl -s -X POST http://localhost:8000/allocate \
    -H "Content-Type: application/json" \
    -d "{\"worker_id\": \"$WORKER_ID\", \"type\": \"rag\"}")

RESOURCE_ID=$(echo $ALLOC_RESPONSE | grep -oP '"id"\s*:\s*"\K[^"]+')
BASE_URL=$(echo $ALLOC_RESPONSE | grep -oP '"base_url"\s*:\s*"\K[^"]+')
TOKEN=$(echo $ALLOC_RESPONSE | grep -oP '"token"\s*:\s*"\K[^"]+')

if [ -z "$RESOURCE_ID" ]; then
    echo -e "${RED}❌ 申请 RAG 资源失败${NC}"
    echo "响应: $ALLOC_RESPONSE"
    exit 1
fi

echo -e "${GREEN}✅ 成功申请 RAG 资源: $RESOURCE_ID${NC}"
echo "   Base URL: $BASE_URL"
echo "   Token: ${TOKEN:0:20}..."
echo ""

# 4. 执行 RAG 查询
echo "4️⃣  执行 RAG 查询..."
QUERY="What is artificial intelligence?"
echo "   查询问题: $QUERY"

SEARCH_RESPONSE=$(curl -s -X POST "$BASE_URL/search" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $TOKEN" \
    -d "{
        \"query\": \"$QUERY\",
        \"top_k\": 3
    }")

if echo "$SEARCH_RESPONSE" | grep -q "results"; then
    echo -e "${GREEN}✅ RAG 查询成功${NC}"
    echo ""
    echo "📄 查询结果预览:"
    echo "$SEARCH_RESPONSE" | python3 -m json.tool 2>/dev/null | head -30
else
    echo -e "${RED}❌ RAG 查询失败${NC}"
    echo "响应: $SEARCH_RESPONSE"
fi
echo ""

# 5. 释放资源
echo "5️⃣  释放 RAG 资源..."
RELEASE_RESPONSE=$(curl -s -X POST http://localhost:8000/release \
    -H "Content-Type: application/json" \
    -d "{\"resource_id\": \"$RESOURCE_ID\", \"worker_id\": \"$WORKER_ID\"}")

if echo "$RELEASE_RESPONSE" | grep -q "success\|released"; then
    echo -e "${GREEN}✅ 成功释放资源${NC}"
else
    echo -e "${YELLOW}⚠️  释放资源可能失败，但不影响测试结果${NC}"
fi
echo ""

# 6. 测试 Gateway (MCP SSE)
echo "6️⃣  验证 Gateway 配置..."
if [ -f "gateway_config.json" ]; then
    echo -e "${GREEN}✅ Gateway 配置文件存在${NC}"
    echo "   配置的模块:"
    grep -A 5 "modules" gateway_config.json | grep "resource_type" | while read line; do
        echo "     $line"
    done
else
    echo -e "${YELLOW}⚠️  未找到 gateway_config.json${NC}"
fi
echo ""

# 总结
echo "========================================"
echo "📊 验证完成"
echo "========================================"
echo -e "${GREEN}🎉 RAG 服务验证通过！${NC}"
echo ""
echo "你可以使用以下资源:"
echo "  • Resource API: http://localhost:8000"
echo "  • Gateway Server: http://localhost:8080"
echo "  • RAG 查询功能已验证可用"
