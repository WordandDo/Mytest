#!/bin/bash
# run_all_benchmarks.sh
# 按顺序运行所有 RAG 基准测试

set -e  # Exit on error

echo "=========================================="
echo "Running All RAG Benchmarks"
echo "=========================================="
echo ""

# 1. No Tool Baseline (最快，不需要 Gateway)
echo "🔹 [1/4] Running No-Tool Baseline..."
./benchmark_no_tool.sh
echo ""
echo "✅ No-Tool Baseline completed"
echo ""

# 2. Dense Only
echo "🔹 [2/4] Running Dense-Only Benchmark..."
./benchmark_dense.sh
echo ""
echo "✅ Dense-Only completed"
echo ""

# 3. Sparse Only
echo "🔹 [3/4] Running Sparse-Only Benchmark..."
./benchmark_sparse.sh
echo ""
echo "✅ Sparse-Only completed"
echo ""

# 4. Hybrid
echo "🔹 [4/4] Running Hybrid Benchmark..."
./benchmark_hybrid.sh
echo ""
echo "✅ Hybrid completed"
echo ""

echo "=========================================="
echo "All Benchmarks Completed!"
echo "=========================================="
echo ""
echo "Results saved in:"
echo "  - results/benchmark_no_tool/"
echo "  - results/benchmark_dense_only/"
echo "  - results/benchmark_sparse_only/"
echo "  - results/benchmark_hybrid/"
echo ""
echo "To compare results, check each directory for metrics files."
