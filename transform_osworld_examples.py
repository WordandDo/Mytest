import json

def transform_osworld_examples(input_file, output_file):
    """
    Transform OSWorld examples from old format to new format with resource_configs.
    
    Old format had config and evaluator at top level.
    New format wraps them in resource_configs.vm.content.
    """
    transformed_items = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                
                # Extract fields that stay at top level
                item_id = data.get('id')
                question = data.get('question')
                answer = data.get('answer', '')
                
                # Extract config and evaluator for moving to resource_configs
                config = data.get('config', [])
                evaluator = data.get('evaluator', {})
                
                # Create new format with resource_configs
                new_item = {
                    'id': item_id,
                    'question': question,
                    'answer': answer,
                    'resource_configs': {
                        'vm': {
                            'init_type': 'osworld_task_spec',
                            'content': {
                                'config': config,
                                'evaluator': evaluator
                            }
                        }
                    }
                }
                
                transformed_items.append(new_item)
                
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON on line {line_num}: {e}")
                continue
    
    # Write transformed items to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in transformed_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Transformed {len(transformed_items)} items from {input_file} to {output_file}")

if __name__ == "__main__":
    input_path = "src/data/osworld_examples.jsonl"
    output_path = "src/data/osworld_examples_transformed.jsonl"
    transform_osworld_examples(input_path, output_path)