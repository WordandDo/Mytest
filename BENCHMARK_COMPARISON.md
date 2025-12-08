# Benchmark Configurations Comparison

## Quick Reference Table

| Benchmark Script | Prompt Type | Gateway Config | Tools Available | Primary Use Case |
|-----------------|-------------|----------------|-----------------|------------------|
| `benchmark_dense.sh` | `generic` | `gateway_config_rag_dense_only.json` | Dense retrieval (E5) | Semantic search, concept matching |
| `benchmark_sparse.sh` | `sparse` | `gateway_config_rag_sparse_only.json` | Sparse retrieval (BM25) | Keyword search, exact matching |
| `benchmark_hybrid.sh` | `hybrid` | `gateway_config_rag_hybrid.json` | Dense + Sparse | Best of both worlds |
| `benchmark_no_tool.sh` | `no_tool` | `gateway_config_rag_no_tool.json` | None | Baseline (pure LLM) |

## Prompt Type Details

### `generic` Prompt
**Philosophy**: General-purpose retrieval with basic guidance

**Key Instructions**:
- Use tools to solve problems
- Break down complex problems into logical steps
- Use ONE tool at a time
- Verify findings through different approaches

**Best For**: Dense-only retrieval where tool choice is obvious

---

### `sparse` Prompt
**Philosophy**: Keyword-focused retrieval strategy

**Key Instructions**:
- **Focus on Keywords**: BM25 relies on exact keyword matching
- **Query Formulation**: Use specific keywords expected in target documents
- Most effective for: specific entities, precise terminology, IDs, exact phrases

**Best For**: When you need to find exact matches or specific details

---

### `hybrid` Prompt
**Philosophy**: Intelligent tool selection based on query nature

**Key Instructions**:

**When to use Sparse Retrieval (BM25)**:
- Exact names, IDs, codes, specific numbers
- Rare technical terms or jargon
- Verify exact presence of a phrase
- When Dense retrieval returns hallucinated/irrelevant matches

**When to use Dense Retrieval (Semantic)**:
- Concepts, summaries, explanations
- Don't know exact keywords but know the meaning
- Exploring a topic broadly

**Strategy**: Try one method, if it fails, try the other

**Best For**: Complex scenarios requiring flexible retrieval strategies

---

### `no_tool` Prompt
**Philosophy**: Pure language model capabilities without external retrieval

**Key Instructions**:
- Answer using ONLY your own knowledge
- Do NOT use any external tools or retrieval functions

**Best For**: Establishing a baseline to measure RAG improvements

---

## Expected Performance Characteristics

### Dense Retrieval (E5/Contriever)
- ✅ Strong at semantic understanding
- ✅ Handles synonyms and paraphrasing well
- ✅ Good for conceptual questions
- ❌ May miss exact keyword matches
- ❌ Can retrieve semantically similar but factually wrong information

### Sparse Retrieval (BM25)
- ✅ Excellent at exact keyword matching
- ✅ Fast and efficient
- ✅ Transparent (keyword-based)
- ❌ Misses semantic relationships
- ❌ Struggles with synonyms and paraphrasing
- ❌ Requires knowing the right keywords

### Hybrid Retrieval
- ✅ Combines strengths of both methods
- ✅ Most flexible
- ✅ Can adapt to different query types
- ⚠️ Requires model to choose correctly
- ⚠️ May have higher latency (multiple retrievals)

### No Tool (Pure LLM)
- ✅ No retrieval latency
- ✅ Uses model's full parametric knowledge
- ❌ Limited to training data (knowledge cutoff)
- ❌ Can hallucinate when uncertain
- ❌ No access to specific documents/data

---

## Experimental Design Recommendations

### Baseline Comparison
```bash
# 1. Run baseline (no tools)
./benchmark_no_tool.sh

# 2. Run single-method benchmarks
./benchmark_dense.sh
./benchmark_sparse.sh

# 3. Run hybrid
./benchmark_hybrid.sh
```

### Question Type Analysis
After running benchmarks, analyze results by question type:
- **Factual questions**: Sparse likely wins
- **Conceptual questions**: Dense likely wins
- **Mixed questions**: Hybrid should perform best

### Dataset Considerations
- **Bamboogle**: Mix of factual and conceptual questions
- **Custom datasets**: Adjust `DATA_PATH` in scripts

---

## Port Configuration

All benchmarks use port 8080 for the MCP gateway:
```bash
lsof -ti:8080 | xargs kill -9 2>/dev/null  # Cleanup before starting
python src/mcp_server/main.py --config <config>.json --port 8080 &
```

If port 8080 is in use, modify the `--port` parameter in each benchmark script.

---

## Resource Requirements

| Benchmark | Gateway Required | Resource API Required | Estimated Time (10 rollouts) |
|-----------|------------------|----------------------|------------------------------|
| Dense | ✅ Yes | ✅ Yes | ~5-10 minutes |
| Sparse | ✅ Yes | ✅ Yes | ~3-7 minutes |
| Hybrid | ✅ Yes | ✅ Yes | ~7-15 minutes |
| No Tool | ❌ No | ❌ No | ~2-5 minutes |

*Times vary based on model speed and question complexity*

---

## Troubleshooting

### Gateway Not Starting
```bash
# Check if port is in use
lsof -i:8080

# Manually kill processes
kill -9 <PID>

# Check gateway logs
tail -f gateway.log  # if logging is enabled
```

### No Tools Available in Hybrid Mode
- Verify `gateway_config_rag_hybrid.json` includes both `rag_query` and `rag_query_sparse`
- Check MCP server is properly initialized
- Review environment logs for tool registration

### Incorrect Prompt Type
- Verify `PROMPT_TYPE` environment variable is set correctly
- Check `src/run_parallel_rollout.py` receives `--prompt_type` argument
- Confirm `HttpMCPRagEnv` is using the correct prompt in `get_system_prompt()`

---

## Advanced Usage

### Custom Prompt Types
To add a new prompt type:

1. Add prompt constant in `src/envs/http_mcp_rag_env.py`:
```python
SYSTEM_PROMPT_CUSTOM = """Your custom prompt here"""
```

2. Update `get_system_prompt()` method:
```python
elif prompt_type == "custom":
    prompt = SYSTEM_PROMPT_CUSTOM
```

3. Update `run_parallel_rollout.py` choices:
```python
choices=["generic", "no_tool", "sparse", "hybrid", "custom"]
```

### Custom Gateway Configuration
Create a new gateway config JSON:
```json
{
  "server_name": "My Custom Gateway",
  "port": 8080,
  "debug": true,
  "modules": [
    {
      "resource_type": "system",
      "tool_groups": ["system_resource"]
    },
    {
      "resource_type": "rag_hybrid",
      "tool_groups": ["rag_query", "rag_query_sparse", "my_custom_tool"]
    }
  ]
}
```

---

## Related Files

- **Environment**: `src/envs/http_mcp_rag_env.py`
- **Rollout Script**: `src/run_parallel_rollout.py`
- **Shared Benchmark Script**: `run_rag_benchmark.sh`
- **Gateway Configs**: `gateway_config_rag_*.json`
- **Benchmark Scripts**: `benchmark_*.sh`
