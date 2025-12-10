#!/usr/bin/env python3
"""从一个 JSON 数组或 JSONL 文件中随机抽样若干项并写出到目标文件。

用法示例:
  python3 sample_seed_entities.py --input example_seed_entities.json --output sample_entities_100.json --count 100 --seed 42
"""
import argparse
import json
import os
import random
import sys


def load_items(path):
    with open(path, 'r', encoding='utf-8') as f:
        start = f.read(2048)
        if not start:
            return []
        stripped = start.lstrip()
        # 判断是 JSON 数组（以 [ 开头）还是逐行的 JSONL/纯文本
        if stripped.startswith('['):
            f.seek(0)
            return json.load(f)
        else:
            f.seek(0)
            items = [line.strip() for line in f if line.strip()]
            # 如果每行是 JSON 字符串（带引号），尝试解析每行为 JSON，否则按文本处理
            parsed = []
            for line in items:
                try:
                    parsed.append(json.loads(line))
                except Exception:
                    parsed.append(line)
            return parsed


def write_items_as_json(items, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description='从 JSON/JSONL 文件随机抽样若干项')
    parser.add_argument('--input', '-i', required=True, help='输入文件路径，支持 JSON 数组 或 每行一个条目的文件')
    parser.add_argument('--output', '-o', required=True, help='输出文件路径（JSON 数组）')
    parser.add_argument('--count', '-n', type=int, default=100, help='要抽样的数量，默认 100')
    parser.add_argument('--seed', type=int, default=None, help='随机种子（可选）')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    items = load_items(args.input)
    if not items:
        print('输入文件未发现可用条目：', args.input, file=sys.stderr)
        sys.exit(2)

    k = args.count
    total = len(items)
    if k <= 0:
        print('抽样数量必须大于 0', file=sys.stderr)
        sys.exit(2)

    if k >= total:
        print(f'要求 {k} 项，但输入仅有 {total} 项；将返回全部项目（不重复）')
        sampled = items.copy()
    else:
        sampled = random.sample(items, k)

    write_items_as_json(sampled, args.output)
    print(f'已写出 {len(sampled)} 项到: {args.output}')


if __name__ == '__main__':
    main()
