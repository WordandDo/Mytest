#!/usr/bin/env python3
"""
Script to convert HotPotQA.jsonl to benchmark format.
Converts from complex HotPotQA format with contexts to simplified benchmark format.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any


def extract_answer(answers_objects: List[Dict]) -> str:
    """
    Extract the main answer from answers_objects.

    Args:
        answers_objects: List of answer objects from HotPotQA format

    Returns:
        Answer string (first span if available)
    """
    if not answers_objects or len(answers_objects) == 0:
        return ""

    first_answer = answers_objects[0]

    # Try to get spans first (most common)
    if 'spans' in first_answer and first_answer['spans']:
        return first_answer['spans'][0]

    # Fallback to number if no spans
    if 'number' in first_answer and first_answer['number']:
        return str(first_answer['number'])

    return ""


def format_contexts_for_question(contexts: List[Dict]) -> str:
    """
    Format contexts as a readable string for the question.

    Args:
        contexts: List of context objects

    Returns:
        Formatted context string
    """
    formatted = []
    for ctx in contexts:
        title = ctx.get('title', '')
        text = ctx.get('paragraph_text', '')
        formatted.append(f"[{title}] {text}")

    return "\n\n".join(formatted)


def convert_hotpotqa_to_benchmark(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a single HotPotQA entry to benchmark format.

    HotPotQA format:
    {
        "dataset": "hotpotqa",
        "question_id": "...",
        "question_text": "...",
        "level": "hard",
        "type": "bridge",
        "answers_objects": [...],
        "contexts": [...]
    }

    Benchmark format:
    {
        "id": "...",
        "question": "...",
        "answer": "...",
        "metadata": {...}
    }
    """
    # Extract answer
    answer = extract_answer(entry.get('answers_objects', []))

    # Format contexts as part of the question
    contexts_text = format_contexts_for_question(entry.get('contexts', []))

    # Combine question with contexts
    full_question = f"{entry.get('question_text', '')}\n\nContexts:\n{contexts_text}"

    # Build benchmark item
    benchmark_item = {
        'id': entry.get('question_id', ''),
        'question': full_question,
        'answer': answer,
        'metadata': {
            'dataset': entry.get('dataset', 'hotpotqa'),
            'level': entry.get('level', ''),
            'type': entry.get('type', ''),
            'original_question': entry.get('question_text', ''),
            'contexts': entry.get('contexts', []),
            'all_answer_spans': [span for ans_obj in entry.get('answers_objects', [])
                                for span in ans_obj.get('spans', [])]
        }
    }

    return benchmark_item


def convert_hotpotqa(input_file: str, output_file: str, validate_only: bool = False,
                    include_contexts_in_question: bool = True):
    """
    Convert HotPotQA.jsonl to benchmark format.

    Args:
        input_file: Path to input JSONL file (HotPotQA format)
        output_file: Path to output JSONL file (benchmark format)
        validate_only: If True, only validate without writing output
        include_contexts_in_question: If True, include contexts in the question field
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    converted_entries = []
    error_count = 0
    total_count = 0

    print(f"Reading from: {input_file}")
    print(f"Converting HotPotQA format to Benchmark format...")

    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            total_count += 1

            try:
                entry = json.loads(line.strip())

                # Convert to benchmark format
                benchmark_item = convert_hotpotqa_to_benchmark(entry)

                # If not including contexts in question, simplify
                if not include_contexts_in_question:
                    benchmark_item['question'] = entry.get('question_text', '')

                converted_entries.append(benchmark_item)

            except json.JSONDecodeError as e:
                error_count += 1
                print(f"Line {line_num}: JSON decode error - {e}")
            except Exception as e:
                error_count += 1
                print(f"Line {line_num}: Conversion error - {e}")

    print(f"\nProcessing complete:")
    print(f"  Total entries: {total_count}")
    print(f"  Converted entries: {len(converted_entries)}")
    print(f"  Errors: {error_count}")

    if validate_only:
        print("\nValidation-only mode. No output file written.")
        if converted_entries:
            print("\nSample converted entry (first entry):")
            sample = converted_entries[0].copy()
            # Truncate long question for display
            if len(sample['question']) > 200:
                sample['question'] = sample['question'][:200] + "..."
            print(json.dumps(sample, indent=2, ensure_ascii=False))
        return

    # Write output
    print(f"\nWriting to: {output_file}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in converted_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"Successfully wrote {len(converted_entries)} entries to {output_file}")

    # Print sample entry
    if converted_entries:
        print("\nSample converted entry (first entry):")
        sample = converted_entries[0].copy()
        # Truncate for display
        if len(sample['question']) > 200:
            sample['question'] = sample['question'][:200] + "..."
        if 'metadata' in sample and 'contexts' in sample['metadata']:
            sample['metadata']['contexts'] = f"[{len(sample['metadata']['contexts'])} contexts]"
        print(json.dumps(sample, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(
        description='Convert HotPotQA.jsonl to benchmark format',
        epilog='Example: python3 convert_hotpotqa_to_benchmark.py -i src/data/HotPotQA.jsonl -o src/data/HotPotQA_benchmark.jsonl'
    )
    parser.add_argument(
        '--input', '-i',
        default='src/data/HotPotQA.jsonl',
        help='Input JSONL file in HotPotQA format (default: src/data/HotPotQA.jsonl)'
    )
    parser.add_argument(
        '--output', '-o',
        default='src/data/HotPotQA_benchmark.jsonl',
        help='Output JSONL file in benchmark format (default: src/data/HotPotQA_benchmark.jsonl)'
    )
    parser.add_argument(
        '--validate-only', '-v',
        action='store_true',
        help='Only validate and show sample without writing output'
    )
    parser.add_argument(
        '--no-contexts',
        action='store_true',
        help='Do not include contexts in the question field (contexts stored in metadata only)'
    )

    args = parser.parse_args()

    try:
        convert_hotpotqa(
            args.input,
            args.output,
            args.validate_only,
            include_contexts_in_question=not args.no_contexts
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
