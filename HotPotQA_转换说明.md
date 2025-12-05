# HotPotQA 数据转换说明

## 概述

成功将 `src/data/HotPotQA.jsonl` 从原始的 HotPotQA 格式转换为 Benchmark 类所需的标准格式。

## 转换脚本

**脚本文件**: `convert_hotpotqa_to_benchmark.py`

### 基本用法

```bash
# 默认转换 (包含上下文在问题中)
python3 convert_hotpotqa_to_benchmark.py

# 只验证不输出
python3 convert_hotpotqa_to_benchmark.py --validate-only

# 不包含上下文在问题字段中 (仅存在 metadata)
python3 convert_hotpotqa_to_benchmark.py --no-contexts

# 自定义输入输出路径
python3 convert_hotpotqa_to_benchmark.py -i input.jsonl -o output.jsonl
```

## 格式对比

### 原始 HotPotQA 格式

```json
{
  "dataset": "hotpotqa",
  "question_id": "5a77666555429966f1a36d1f",
  "question_text": "What is the nationality of...",
  "level": "hard",
  "type": "bridge",
  "answers_objects": [
    {
      "number": "",
      "date": {"day": "", "month": "", "year": ""},
      "spans": ["American"]
    }
  ],
  "contexts": [
    {
      "idx": 0,
      "title": "Oliver Fricker",
      "paragraph_text": "...",
      "is_supporting": true
    }
  ]
}
```

### 转换后 Benchmark 格式

```json
{
  "id": "5a77666555429966f1a36d1f",
  "question": "What is the nationality of...\n\nContexts:\n[Title1] paragraph1\n\n[Title2] paragraph2...",
  "answer": "American",
  "metadata": {
    "dataset": "hotpotqa",
    "level": "hard",
    "type": "bridge",
    "original_question": "What is the nationality of...",
    "contexts": [...],
    "all_answer_spans": ["American"]
  }
}
```

## 关键转换逻辑

1. **ID映射**: `question_id` → `id`
2. **问题构建**: `question_text` + 格式化的 `contexts` → `question`
3. **答案提取**: `answers_objects[0]['spans'][0]` → `answer` (字符串)
4. **元数据保留**: 所有原始信息保存在 `metadata` 中

## 数据统计

- **总条目数**: 500
- **全部为 hard 难度**
- **类型分布**:
  - bridge: 412 (82.4%)
  - comparison: 88 (17.6%)
- **答案长度**: 平均 14.4 字符

## 在 Benchmark 类中使用

```python
from benchmark import create_benchmark

# 加载数据
benchmark = create_benchmark(
    data_path='src/data/HotPotQA_benchmark.jsonl',
    name='HotPotQA',
    description='HotPotQA多跳问答数据集'
)

# 获取数据
items = benchmark.get_items()

# 评估
predictions = {item.id: "prediction_text" for item in items}
results = benchmark.evaluate(predictions, metric='exact_match')
```

## 文件清单

1. `convert_hotpotqa_to_benchmark.py` - 转换脚本
2. `verify_hotpotqa_conversion.py` - 验证脚本
3. `src/data/HotPotQA.jsonl` - 原始数据 (3.2MB)
4. `src/data/HotPotQA_benchmark.jsonl` - 转换后数据 (8.9MB)

## 注意事项

- 转换后的 `question` 字段包含了完整的上下文信息，文件会变大
- 如果只需要原始问题，使用 `metadata['original_question']`
- 原始的 `contexts` 数组完整保留在 `metadata` 中
- 支持性段落可通过 `is_supporting` 标识

---

转换完成时间: 2025-12-05
