# RAG Benchmark Guide

## Overview
This guide explains how to run different RAG benchmark experiments with various retrieval strategies and prompt types.

## Available Benchmark Scripts

### 1. Dense-Only RAG (`benchmark_dense.sh`)
- **Retrieval Method**: Dense vector retrieval only (E5/Contriever)
- **Prompt Type**: `generic`
- **Gateway Config**: `gateway_config_rag_dense_only.json`
- **Output Directory**: `results/benchmark_dense_only`

```bash
./benchmark_dense.sh
```

### 2. Sparse-Only RAG (`benchmark_sparse.sh`)
- **Retrieval Method**: Sparse keyword retrieval only (BM25)
- **Prompt Type**: `sparse` (keyword-focused instructions)
- **Gateway Config**: `gateway_config_rag_sparse_only.json`
- **Output Directory**: `results/benchmark_sparse_only`

```bash
./benchmark_sparse.sh
```

### 3. Hybrid RAG (`benchmark_hybrid.sh`)
- **Retrieval Method**: Both Dense and Sparse retrieval
- **Prompt Type**: `hybrid` (instructions for choosing between methods)
- **Gateway Config**: `gateway_config_rag_hybrid.json`
- **Output Directory**: `results/benchmark_hybrid`

```bash
./benchmark_hybrid.sh
```

### 4. No Tool (Pure LLM) (`benchmark_no_tool.sh`)
- **Retrieval Method**: None (no RAG tools)
- **Prompt Type**: `no_tool` (relies only on model's knowledge)
- **Gateway Config**: `gateway_config_rag_no_tool.json`
- **Output Directory**: `results/benchmark_no_tool`
- **Note**: This script does NOT start a gateway, as no tools are needed

```bash
./benchmark_no_tool.sh
```

## Prompt Types Explained

### `generic`
- General-purpose RAG prompt
- Suitable for any retrieval tool
- No specific guidance on tool selection

### `no_tool`
- Instructs the model to answer without using any tools
- Relies solely on the model's internal knowledge
- No tool descriptions are added to the prompt

### `sparse`
- Optimized for keyword-based (BM25) retrieval
- Emphasizes exact keyword matching
- Provides guidance on query formulation for sparse retrieval

### `hybrid`
- Designed for environments with both Dense and Sparse retrieval
- Provides clear guidelines on when to use each method:
  - **Sparse**: For exact names, IDs, codes, specific numbers
  - **Dense**: For concepts, summaries, explanations

## Configuration Files

### Gateway Configs

| File | Description | Tool Groups |
|------|-------------|-------------|
| `gateway_config_rag_dense_only.json` | Dense retrieval only | `rag_query` |
| `gateway_config_rag_sparse_only.json` | Sparse retrieval only | `rag_query_sparse` |
| `gateway_config_rag_hybrid.json` | Both retrieval methods | `rag_query`, `rag_query_sparse` |
| `gateway_config_rag_no_tool.json` | No RAG tools | (system only) |

## Customizing Benchmarks

### Modifying Parameters

Each benchmark script exports environment variables that can be customized:

```bash
# In benchmark_*.sh, modify these variables:
export OUTPUT_DIR="results/my_custom_test"
export DATA_PATH="src/data/my_dataset.json"
export NUM_ROLLOUTS=20
export GATEWAY_CONFIG_PATH="my_custom_config.json"
export PROMPT_TYPE="hybrid"
```

### Using `run_rag_benchmark.sh` Directly

You can also set environment variables and call `run_rag_benchmark.sh` directly:

```bash
export OUTPUT_DIR="results/custom_test"
export DATA_PATH="src/data/bamboogle.json"
export NUM_ROLLOUTS=10
export GATEWAY_CONFIG_PATH="gateway_config_rag_hybrid.json"
export PROMPT_TYPE="hybrid"
export MODEL_NAME="openai/gpt-oss-120b"
export MAX_TURNS=15

./run_rag_benchmark.sh
```

## Implementation Details

### Code Changes

1. **`src/envs/http_mcp_rag_env.py`**:
   - Added four system prompt templates: `SYSTEM_PROMPT_GENERIC`, `SYSTEM_PROMPT_NO_TOOLS`, `SYSTEM_PROMPT_SPARSE`, `SYSTEM_PROMPT_HYBRID`
   - Updated `get_system_prompt()` method to support `prompt_type` parameter
   - Added `self.prompt_type` instance variable to store the prompt type

2. **`src/run_parallel_rollout.py`**:
   - Added `--prompt_type` argument with choices: `generic`, `no_tool`, `sparse`, `hybrid`
   - Passed `prompt_type` to environment via `env_kwargs`

3. **`run_rag_benchmark.sh`**:
   - Added `PROMPT_TYPE` environment variable with default value `generic`
   - Added `--prompt_type "$PROMPT_TYPE"` to the Python command

4. **All benchmark scripts**:
   - Set appropriate `PROMPT_TYPE` for each benchmark mode

## Running All Benchmarks

To run all four benchmarks sequentially:

```bash
./benchmark_dense.sh
./benchmark_sparse.sh
./benchmark_hybrid.sh
./benchmark_no_tool.sh
```

## Evaluation Metrics

All benchmarks use two evaluation metrics by default:
- `exact_match`: Exact string matching
- `f1_score`: Token-level F1 score

## Output Structure

Each benchmark creates a results directory with:
- Detailed rollout logs
- Evaluation metrics
- Task completion statistics

```
results/
├── benchmark_dense_only/
├── benchmark_sparse_only/
├── benchmark_hybrid/
└── benchmark_no_tool/
```

## Notes

- The `no_tool` benchmark does not require a running MCP gateway
- Other benchmarks automatically start and stop the gateway
- Gateway runs on port 8080 by default
- All benchmarks use the same dataset specified in `DATA_PATH`
