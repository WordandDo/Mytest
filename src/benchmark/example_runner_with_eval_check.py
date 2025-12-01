#!/usr/bin/env python3
"""
Example runner showing how to check for evaluator configuration 
before calling evaluate_task to avoid unnecessary network calls.
"""

import json
from typing import Dict, Any, Optional
from benchmark import Benchmark, BenchmarkItem

def has_evaluator(task: BenchmarkItem) -> bool:
    """
    Check if a task has an evaluator configuration.
    
    Args:
        task: BenchmarkItem to check
        
    Returns:
        bool: True if task has evaluator, False otherwise
    """
    try:
        # Check if task has metadata with resource_configs
        if not task.metadata or "resource_configs" not in task.metadata:
            return False
            
        # Get VM configuration
        vm_config = task.metadata.get("resource_configs", {}).get("vm", {})
        if not vm_config:
            return False
            
        # Get content which may contain evaluator
        content = vm_config.get("content", {})
        
        # Content might be a JSON string or dict
        if isinstance(content, str) and content.strip().startswith("{"):
            content = json.loads(content)
            
        # Check if evaluator exists in content
        if isinstance(content, dict) and "evaluator" in content:
            evaluator = content["evaluator"]
            # Additional check: make sure evaluator is not empty
            if evaluator and isinstance(evaluator, dict) and evaluator.get("func"):
                return True
                
    except Exception as e:
        print(f"Error checking evaluator for task {task.id}: {e}")
        
    return False

def run_task_with_selective_evaluation(env, task: BenchmarkItem, worker_id: str) -> Dict[str, Any]:
    """
    Run a task and conditionally execute evaluation based on presence of evaluator.
    
    Args:
        env: Environment instance
        task: BenchmarkItem to run
        worker_id: Worker identifier
        
    Returns:
        Dict containing task results and evaluation (if applicable)
    """
    result = {
        "task_id": task.id,
        "question": task.question,
        "has_evaluator": False,
        "evaluation_score": "0.0"
    }
    
    # Check if task has evaluator
    if has_evaluator(task):
        result["has_evaluator"] = True
        try:
            print(f"Executing evaluation for {task.id}...")
            eval_result = env.execute_tool("evaluate_task", {"worker_id": worker_id})
            
            # Extract score from result
            if isinstance(eval_result, str):
                try:
                    # Try to parse as float
                    float(eval_result)
                    result["evaluation_score"] = eval_result
                except ValueError:
                    # If it's not a simple float, it might be a JSON response
                    try:
                        eval_data = json.loads(eval_result)
                        if isinstance(eval_data, dict) and "score" in eval_data:
                            result["evaluation_score"] = str(eval_data["score"])
                    except:
                        # Fallback to original result
                        result["evaluation_score"] = eval_result
        except Exception as e:
            print(f"Eval error for task {task.id}: {e}")
            result["evaluation_score"] = "0.0"
    else:
        print(f"Skipping evaluation for {task.id} (No evaluator found)")
        # For tasks without evaluator, keep default score "0.0" or could set to "N/A"
        
    return result

# Example usage
if __name__ == "__main__":
    # Load benchmark
    benchmark = Benchmark(data_path="../data/osworld_examples_transformed.jsonl")
    
    print(f"Loaded {len(benchmark.items)} benchmark items")
    
    # Example processing (without actual environment for simplicity)
    for item in benchmark.items[:1]:  # Just first item for demo
        print(f"\nTask ID: {item.id}")
        print(f"Question: {item.question}")
        print(f"Has evaluator: {has_evaluator(item)}")
        
        # In a real scenario, you would:
        # 1. Initialize environment
        # 2. Execute the task using the environment
        # 3. Conditionally call evaluate_task based on has_evaluator check
        # result = run_task_with_selective_evaluation(env, item, worker_id)