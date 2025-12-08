#!/bin/bash
# verify_setup.sh - 验证所有文件是否正确设置

echo "=========================================="
echo "验证 RAG Benchmark 设置"
echo "=========================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查计数
TOTAL=0
PASSED=0
FAILED=0

check_file() {
    TOTAL=$((TOTAL + 1))
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} $1"
        PASSED=$((PASSED + 1))
    else
        echo -e "${RED}✗${NC} $1 (缺失)"
        FAILED=$((FAILED + 1))
    fi
}

check_executable() {
    TOTAL=$((TOTAL + 1))
    if [ -x "$1" ]; then
        echo -e "${GREEN}✓${NC} $1 (可执行)"
        PASSED=$((PASSED + 1))
    else
        echo -e "${YELLOW}⚠${NC} $1 (不可执行，尝试添加权限)"
        chmod +x "$1" 2>/dev/null
        if [ -x "$1" ]; then
            echo -e "${GREEN}  → 已修复${NC}"
            PASSED=$((PASSED + 1))
        else
            echo -e "${RED}  → 修复失败${NC}"
            FAILED=$((FAILED + 1))
        fi
    fi
}

echo "1. 检查 Benchmark 脚本..."
check_executable "benchmark_dense.sh"
check_executable "benchmark_sparse.sh"
check_executable "benchmark_hybrid.sh"
check_executable "benchmark_no_tool.sh"
check_executable "run_all_benchmarks.sh"
echo ""

echo "2. 检查 Gateway 配置文件..."
check_file "gateway_config_rag_dense_only.json"
check_file "gateway_config_rag_sparse_only.json"
check_file "gateway_config_rag_hybrid.json"
check_file "gateway_config_rag_no_tool.json"
echo ""

echo "3. 检查文档文件..."
check_file "BENCHMARK_GUIDE.md"
check_file "BENCHMARK_COMPARISON.md"
check_file "IMPLEMENTATION_SUMMARY.md"
check_file "QUICKSTART.md"
echo ""

echo "4. 检查核心 Python 文件..."
check_file "src/envs/http_mcp_rag_env.py"
check_file "src/run_parallel_rollout.py"
check_file "run_rag_benchmark.sh"
echo ""

echo "5. 检查数据文件..."
if [ -f "src/data/bamboogle.json" ]; then
    echo -e "${GREEN}✓${NC} src/data/bamboogle.json"
    PASSED=$((PASSED + 1))
else
    echo -e "${YELLOW}⚠${NC} src/data/bamboogle.json (缺失，可能需要下载或使用其他数据集)"
fi
TOTAL=$((TOTAL + 1))
echo ""

echo "6. 检查端口 8080 是否可用..."
if lsof -i:8080 > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠${NC} 端口 8080 正在使用中"
    echo "   运行 'lsof -ti:8080 | xargs kill -9' 来清理"
else
    echo -e "${GREEN}✓${NC} 端口 8080 可用"
    PASSED=$((PASSED + 1))
fi
TOTAL=$((TOTAL + 1))
echo ""

echo "=========================================="
echo "验证结果"
echo "=========================================="
echo "总计: $TOTAL"
echo -e "${GREEN}通过: $PASSED${NC}"
if [ $FAILED -gt 0 ]; then
    echo -e "${RED}失败: $FAILED${NC}"
fi
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ 所有检查通过！可以开始运行基准测试。${NC}"
    echo ""
    echo "快速开始："
    echo "  ./benchmark_dense.sh     # Dense-only RAG"
    echo "  ./benchmark_sparse.sh    # Sparse-only RAG"
    echo "  ./benchmark_hybrid.sh    # Hybrid RAG"
    echo "  ./benchmark_no_tool.sh   # No Tool baseline"
    echo "  ./run_all_benchmarks.sh  # 运行所有测试"
else
    echo -e "${RED}✗ 发现 $FAILED 个问题，请检查并修复。${NC}"
    exit 1
fi
