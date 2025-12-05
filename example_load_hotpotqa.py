#!/usr/bin/env python3
"""
示例脚本：如何加载和使用转换后的HotPotQA benchmark数据
"""

import sys
from pathlib import Path

# Add src to path if needed
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from benchmark import create_benchmark


def main():
    """演示如何加载和使用HotPotQA benchmark数据"""

    print("=" * 60)
    print("HotPotQA Benchmark 加载示例")
    print("=" * 60)

    # 1. 加载benchmark数据
    print("\n1. 加载数据...")
    benchmark = create_benchmark(
        data_path="src/data/HotPotQA_benchmark.jsonl",
        name="HotPotQA",
        description="HotPotQA 多跳问答数据集"
    )

    # 2. 查看数据集信息
    print(f"\n2. 数据集信息:")
    print(f"   - 总问题数: {len(benchmark.get_items())}")

    # 3. 查看样本数据
    print(f"\n3. 查看第一个问题:")
    items = benchmark.get_items()
    if items:
        first_item = items[0]
        print(f"   - ID: {first_item.id}")
        print(f"   - 原始问题: {first_item.metadata.get('original_question', '')}")
        print(f"   - 答案: {first_item.answer}")
        print(f"   - 难度: {first_item.metadata.get('level', '')}")
        print(f"   - 类型: {first_item.metadata.get('type', '')}")
        print(f"   - 上下文数量: {len(first_item.metadata.get('contexts', []))}")

        # 显示前2个上下文的标题
        contexts = first_item.metadata.get('contexts', [])
        if contexts:
            print(f"\n   前2个上下文标题:")
            for i, ctx in enumerate(contexts[:2]):
                supporting = "✓" if ctx.get('is_supporting') else "✗"
                print(f"      [{i}] {ctx.get('title', '')} (supporting: {supporting})")

    # 4. 统计数据
    print(f"\n4. 数据统计:")
    items = benchmark.get_items()

    # 按难度统计
    level_counts = {}
    type_counts = {}
    for item in items:
        level = item.metadata.get('level', 'unknown')
        level_counts[level] = level_counts.get(level, 0) + 1

        qtype = item.metadata.get('type', 'unknown')
        type_counts[qtype] = type_counts.get(qtype, 0) + 1

    print(f"   按难度分布:")
    for level, count in sorted(level_counts.items()):
        print(f"      - {level}: {count}")

    print(f"\n   按类型分布:")
    for qtype, count in sorted(type_counts.items()):
        print(f"      - {qtype}: {count}")

    # 5. 模拟评估
    print(f"\n5. 模拟评估示例:")
    print("   (假设我们有一些预测结果)")

    # 创建模拟预测 (实际应用中，这些来自Agent)
    predictions = {}
    for item in items[:3]:  # 只取前3个作为示例
        # 模拟：第1个答对，第2、3个答错
        if item.id == items[0].id:
            predictions[item.id] = item.answer  # 正确答案
        else:
            predictions[item.id] = "Wrong answer"  # 错误答案

    # 执行评估
    results = benchmark.evaluate(predictions, metric="exact_match")

    print(f"\n   评估结果:")
    print(f"   - 平均得分: {results['average_score']:.2f}")
    print(f"   - 总评估数: {results['total_items']}")

    # 显示每个问题的详细结果
    print(f"\n   详细结果 (前3个):")
    for detail in results['details'][:3]:
        status = "✓" if detail['score'] == 1.0 else "✗"
        print(f"      {status} {detail['id']}: {detail['score']}")
        print(f"         预测: {detail['prediction']}")
        print(f"         答案: {detail['ground_truth']}")

    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
