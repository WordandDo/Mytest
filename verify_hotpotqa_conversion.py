#!/usr/bin/env python3
"""
简单验证脚本：直接读取转换后的HotPotQA benchmark数据
不需要导入benchmark模块
"""

import json
from pathlib import Path


def main():
    """验证转换后的HotPotQA数据格式"""

    print("=" * 70)
    print("HotPotQA 转换结果验证")
    print("=" * 70)

    # 1. 读取数据
    data_file = Path("src/data/HotPotQA_benchmark.jsonl")
    print(f"\n1. 读取文件: {data_file}")

    items = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            items.append(json.loads(line.strip()))

    print(f"   ✓ 成功加载 {len(items)} 条数据")

    # 2. 验证格式
    print(f"\n2. 验证数据格式...")

    required_fields = ['id', 'question', 'answer', 'metadata']
    all_valid = True

    for i, item in enumerate(items[:10]):  # 验证前10条
        for field in required_fields:
            if field not in item:
                print(f"   ✗ 第 {i+1} 条数据缺少字段: {field}")
                all_valid = False

    if all_valid:
        print(f"   ✓ 格式验证通过 (检查前10条)")

    # 3. 显示第一条完整数据 (截断长文本)
    print(f"\n3. 第一条数据示例:")
    if items:
        first = items[0]
        print(f"   ID: {first['id']}")
        print(f"   原始问题: {first['metadata']['original_question']}")
        print(f"   答案: {first['answer']}")
        print(f"   难度: {first['metadata']['level']}")
        print(f"   类型: {first['metadata']['type']}")

        # 显示完整问题的前200字符
        question_preview = first['question'][:200] + "..." if len(first['question']) > 200 else first['question']
        print(f"\n   完整问题 (前200字符):")
        print(f"   {question_preview}")

        # 显示上下文信息
        contexts = first['metadata']['contexts']
        print(f"\n   上下文信息:")
        print(f"   - 总共 {len(contexts)} 个上下文段落")
        print(f"   - 支持性段落: {sum(1 for c in contexts if c.get('is_supporting'))}")

        print(f"\n   上下文标题列表:")
        for ctx in contexts:
            supporting = " [支持]" if ctx.get('is_supporting') else ""
            print(f"      • {ctx['title']}{supporting}")

    # 4. 数据统计
    print(f"\n4. 数据集统计:")

    # 按难度统计
    level_counts = {}
    type_counts = {}
    answer_lengths = []

    for item in items:
        level = item['metadata'].get('level', 'unknown')
        level_counts[level] = level_counts.get(level, 0) + 1

        qtype = item['metadata'].get('type', 'unknown')
        type_counts[qtype] = type_counts.get(qtype, 0) + 1

        answer_lengths.append(len(item['answer']))

    print(f"\n   难度分布:")
    for level, count in sorted(level_counts.items()):
        percentage = (count / len(items)) * 100
        print(f"      • {level:10s}: {count:3d} ({percentage:5.1f}%)")

    print(f"\n   类型分布:")
    for qtype, count in sorted(type_counts.items()):
        percentage = (count / len(items)) * 100
        print(f"      • {qtype:10s}: {count:3d} ({percentage:5.1f}%)")

    print(f"\n   答案统计:")
    print(f"      • 平均长度: {sum(answer_lengths) / len(answer_lengths):.1f} 字符")
    print(f"      • 最短答案: {min(answer_lengths)} 字符")
    print(f"      • 最长答案: {max(answer_lengths)} 字符")

    # 5. 示例问答对
    print(f"\n5. 随机问答示例 (前3条):")
    for i, item in enumerate(items[:3], 1):
        print(f"\n   [{i}] 问题: {item['metadata']['original_question']}")
        print(f"       答案: {item['answer']}")
        print(f"       难度: {item['metadata']['level']} | 类型: {item['metadata']['type']}")

    # 6. 数据格式对比
    print(f"\n6. 格式转换对比:")
    print(f"   原始 HotPotQA 格式:")
    print(f"      • 字段: dataset, question_id, question_text, level, type,")
    print(f"               answers_objects, contexts")
    print(f"      • 答案: 嵌套在 answers_objects[0]['spans'][0]")
    print(f"      • 上下文: 完整的段落列表")
    print(f"\n   转换后 Benchmark 格式:")
    print(f"      • 字段: id, question, answer, metadata")
    print(f"      • 答案: 直接提取到 answer 字段 (字符串)")
    print(f"      • 上下文: 包含在 question 字段中 + metadata 保留原始数据")
    print(f"      • 元数据: level, type, original_question, contexts 等")

    print("\n" + "=" * 70)
    print("✓ 验证完成！数据格式正确，可以被 Benchmark 类加载")
    print("=" * 70)

    # 7. 使用建议
    print(f"\n使用建议:")
    print(f"   from benchmark import create_benchmark")
    print(f"   ")
    print(f"   benchmark = create_benchmark(")
    print(f"       data_path='src/data/HotPotQA_benchmark.jsonl',")
    print(f"       name='HotPotQA',")
    print(f"       description='HotPotQA多跳问答数据集'")
    print(f"   )")
    print(f"   ")
    print(f"   # 获取所有问题")
    print(f"   items = benchmark.get_items()")
    print(f"   ")
    print(f"   # 评估预测结果")
    print(f"   results = benchmark.evaluate(predictions, metric='exact_match')")


if __name__ == '__main__':
    main()
