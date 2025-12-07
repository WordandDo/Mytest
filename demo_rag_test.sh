#!/bin/bash
# 快速演示脚本 - 使用小数据集进行测试

echo "=========================================="
echo "RAG 环境测评演示"
echo "=========================================="
echo ""

# 检查数据文件是否存在
if [ ! -f "src/data/rag_demo.jsonl" ]; then
    echo "❌ 错误: 数据文件不存在 src/data/rag_demo.jsonl"
    exit 1
fi

# 显示数据集样本
echo "📄 数据集样本（前3条）:"
head -n 3 src/data/rag_demo.jsonl | python3 -m json.tool 2>/dev/null || head -n 3 src/data/rag_demo.jsonl
echo ""

# 检查 deployment_config.json 中 RAG 是否启用
echo "🔍 检查 RAG 资源配置..."
if grep -q '"rag".*"enabled": true' deployment_config.json; then
    echo "✅ RAG 资源已启用"
else
    echo "❌ 警告: RAG 资源未启用，请检查 deployment_config.json"
    echo "   需要设置: resources.rag.enabled = true"
fi
echo ""

# 显示将要使用的配置
echo "⚙️  测评配置:"
echo "   - 数据集: src/data/rag_demo.jsonl"
echo "   - 并行度: 3 workers"
echo "   - 环境: http_mcp_rag"
echo "   - 测评指标: exact_match, f1_score"
echo "   - 输出目录: results/demo_$(date +%Y%m%d_%H%M%S)"
echo ""

# 询问用户是否继续
read -p "是否继续运行测评? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

# 运行测评
echo ""
echo "🚀 开始运行测评..."
echo ""

DATA_PATH=src/data/rag_demo.jsonl \
NUM_ROLLOUTS=3 \
OUTPUT_DIR=results/demo_$(date +%Y%m%d_%H%M%S) \
./run_rag_benchmark.sh

echo ""
echo "演示完成！"
