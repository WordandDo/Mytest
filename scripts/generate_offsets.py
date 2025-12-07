#!/usr/bin/env python3
"""
生成 JSONL 文件的字节偏移量索引

用法:
    python scripts/generate_offsets.py <jsonl_path>

示例:
    python scripts/generate_offsets.py /home/a1/sdb/wikidata4rag/Search-r1/data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl

输出:
    在同一目录下生成 wiki_dump.offsets 文件
"""
import os
import sys
import struct
from tqdm import tqdm


def generate_offsets(jsonl_path: str):
    """
    扫描 JSONL 文件,生成每一行的字节偏移量索引

    Args:
        jsonl_path: JSONL 文件路径
    """
    if not os.path.exists(jsonl_path):
        print(f"错误: 文件不存在 {jsonl_path}")
        return

    # 确定 .offsets 文件路径
    if jsonl_path.endswith('.jsonl'):
        offset_path = jsonl_path[:-6] + ".offsets"
    else:
        offset_path = jsonl_path + ".offsets"

    print(f"正在扫描文件: {jsonl_path}")
    print(f"输出偏移量文件: {offset_path}")

    # 获取文件大小用于进度条
    file_size = os.path.getsize(jsonl_path)
    print(f"文件大小: {file_size / 1024 / 1024 / 1024:.2f} GB")

    offsets = []

    with open(jsonl_path, 'rb') as f:
        # 使用 tqdm 显示进度(基于字节数)
        with tqdm(total=file_size, unit='B', unit_scale=True, desc="扫描进度") as pbar:
            offset = 0
            while True:
                offsets.append(offset)
                line = f.readline()
                if not line:
                    # 最后一个空行不计入
                    offsets.pop()
                    break
                line_length = len(line)
                offset += line_length
                pbar.update(line_length)

    print(f"\n扫描完成,共 {len(offsets):,} 行")

    # 写入 .offsets 文件 (二进制格式: unsigned long long, 8 bytes per offset)
    print(f"正在写入偏移量文件...")
    with open(offset_path, 'wb') as f:
        # 使用 '<Q' 格式: little-endian, unsigned long long
        packed_data = struct.pack(f'<{len(offsets)}Q', *offsets)
        f.write(packed_data)

    offset_file_size = os.path.getsize(offset_path)
    print(f"✓ 偏移量文件已生成: {offset_path}")
    print(f"  - 索引大小: {offset_file_size / 1024 / 1024:.2f} MB")
    print(f"  - 每行平均: {offset_file_size / len(offsets):.1f} 字节")
    print(f"\n内存节省预估:")
    print(f"  - 原始全量加载: ~{file_size / 1024 / 1024:.0f} MB")
    print(f"  - 懒加载索引: ~{offset_file_size / 1024 / 1024:.0f} MB")
    print(f"  - 节省内存: ~{(file_size - offset_file_size) / 1024 / 1024:.0f} MB")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    jsonl_path = sys.argv[1]
    generate_offsets(jsonl_path)
