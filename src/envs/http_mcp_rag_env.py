# src/envs/http_mcp_rag_env.py
import logging
from typing import Dict, Any, Optional
from .http_mcp_env import HttpMCPEnv

logger = logging.getLogger(__name__)

# =============================================================================
# System Prompts
# =============================================================================

# [修改说明] 移除了 <FINAL_ANSWER> 标签要求，保留了对答案简短性的指导
SYSTEM_PROMPT_GENERIC = """You are a helpful assistant. You need to use tools to solve the problem.
You must use tool to retrieve information to answer and verify. Don't answer by your own knowledge.

## Tool Usage Strategy
1. Break complex problems into logical steps
2. Use ONE tool at a time to gather information
3. Verify findings through different approaches when possible
4. You are encouraged to perform multiple tool-use to get the final answer. If some tool call doesn't output useful response, try other one.
5. You need to use the information you retrieve carefully and accurately, and avoid making incorrect associations or assumptions.
6. After several (more than 10) tool-use, if you still can't get the final answer, you can answer by your own knowledge.

## Answer Strategy
The final answer should only contain the short answer to the question (few words), avoiding unnecessary reasoning content in the final output string.
"""

# [修改说明] 移除了 <FINAL_ANSWER> 标签要求
SYSTEM_PROMPT_NO_TOOLS = """You are a helpful assistant.
You must answer the question using ONLY your own knowledge. Do not use any external tools or retrieval functions.

## Answer Strategy
The final answer should only contain the short answer to the question (few words), avoiding unnecessary reasoning content in the final output string.
"""

# [修改说明] 移除了 <FINAL_ANSWER> 标签要求
SYSTEM_PROMPT_SPARSE = """You are a helpful assistant. You need to use tools to solve the problem.
You must use the sparse retrieval tool to answer and verify. Don't answer by your own knowledge.

## Tool Usage Strategy
1. **Focus on Keywords**: You are using a Sparse Retrieval system (e.g., BM25). It relies on matching exact keywords.
2. **When to use**: This is most effective for finding specific entities, precise terminology, IDs, or exact phrases.
3. **Query Formulation**: Construct queries with the specific keywords you expect to find in the target document. Avoid vague or overly conceptual queries.
4. Break complex problems into logical steps.
5. Verify findings through different approaches when possible.

## Answer Strategy
The final answer should only contain the short answer to the question (few words), avoiding unnecessary reasoning content in the final output string.
"""

# [修改说明] 移除了 <FINAL_ANSWER> 标签要求
SYSTEM_PROMPT_HYBRID = """You are a helpful assistant. You need to use tools to solve the problem.
You have access to a Hybrid Retrieval system consisting of both Sparse (keyword) and Dense (semantic) retrievers.

## Tool Usage Strategy
You must choose the appropriate retrieval method based on the nature of your query.

### 1. When to use Sparse Retrieval (Keyword/BM25)
**Priority**: High for specific details.
**Advantages**: Precise, exact match.
**Use Cases**:
- Searching for **exact names**, **IDs**, **codes**, or **specific numbers**.
- When the query contains rare technical terms or jargon.
- When you need to verify the exact presence of a phrase.
- If Dense retrieval returns hallucinated or irrelevant conceptual matches.

### 2. When to use Dense Retrieval (Semantic/Vector)
**Priority**: High for conceptual questions.
**Advantages**: Understands meaning, handles synonyms and paraphrasing.
**Use Cases**:
- Searching for **concepts**, **summaries**, or **explanations**.
- When you don't know the exact keywords but know the meaning.
- Exploring a topic broadly.

### General Guidelines
1. Break complex problems into logical steps.
2. If one method fails, try the other. For example, if a specific keyword search returns nothing, try a broader semantic search.
3. Use the information you retrieve carefully and accurately.

## Answer Strategy
The final answer should only contain the short answer to the question (few words), avoiding unnecessary reasoning content in the final output string.
"""


class HttpMCPRagEnv(HttpMCPEnv):
    """
    RAG-only environment that inherits from HttpMCPEnv

    This environment is configured to use only RAG resources with a specialized
    prompt for information retrieval tasks. It follows the self-managed lifecycle
    pattern where it explicitly calls setup_rag_session to allocate resources.
    """

    def __init__(self,
                 model_name: str = "gpt-4.1-2025-04-14",
                 parallel_degree: int = 1,
                 **kwargs):

        # Force gateway config to use only RAG resources
        # You can specify a different config file if needed
        if "gateway_config_path" not in kwargs:
            # Use a RAG-specific config file if it exists, otherwise use default
            kwargs["gateway_config_path"] = "gateway_config.json"

        # Store prompt_type for later use in get_system_prompt
        self.prompt_type = kwargs.get("prompt_type", "generic")

        # Initialize parent class
        super().__init__(
            model_name=model_name,
            parallel_degree=parallel_degree,
            **kwargs
        )

        # 减少日志：移除初始化日志
        # logger.info(f"HttpMCPRagEnv initialized for {self.worker_id}")

    @property
    def mode(self) -> str:
        """Override mode to identify this as RAG-specific environment"""
        return "http_mcp_rag"

    def get_system_prompt(self, task_question: Optional[str] = None, **kwargs) -> str:
        """
        Override to use RAG-specific system prompt.

        Args:
            task_question: The current task question (optional)
            **kwargs: Additional keyword arguments.
                      Supports 'prompt_type': 'generic', 'no_tool', 'sparse', 'hybrid'

        Returns:
            The customized system prompt for RAG tasks
        """
        # Use instance prompt_type first, then check kwargs, default to "generic"
        prompt_type = kwargs.get("prompt_type", self.prompt_type)

        # Select base prompt
        if prompt_type == "no_tool":
            prompt = SYSTEM_PROMPT_NO_TOOLS
            # For no_tool, we skip adding tool descriptions
            if task_question:
                prompt += f"\n\n## Current Task\n{task_question}"
            return prompt

        elif prompt_type == "sparse":
            prompt = SYSTEM_PROMPT_SPARSE
        elif prompt_type == "hybrid":
            prompt = SYSTEM_PROMPT_HYBRID
        else:
            prompt = SYSTEM_PROMPT_GENERIC

        # Add tool descriptions (only for tool-enabled prompts)
        tool_descriptions = self.get_tool_descriptions()
        if tool_descriptions:
            prompt += f"\n\n## Available Tools\n{tool_descriptions}"

        # Add task question if provided
        if task_question:
            prompt += f"\n\n## Current Task\n{task_question}"

        return prompt

    def _load_gateway_config(self, config_path: str) -> Dict[str, Any]:
        """
        Override to filter only RAG resources from the config

        This ensures only RAG-related modules are loaded, even if the
        config file contains other resource types.

        NOTE: We must keep 'system' resource type as it provides essential
        tools like allocate_batch_resources and setup_batch_resources.
        Supports both 'rag' and 'rag_hybrid' resource types.
        """
        config = super()._load_gateway_config(config_path)

        # Filter modules to only include RAG resources AND system resources
        # System resources are needed for resource allocation/lifecycle management
        # Support both 'rag' and 'rag_hybrid' resource types
        if "modules" in config:
            original_count = len(config["modules"])
            config["modules"] = [
                module for module in config["modules"]
                if module.get("resource_type") in ["rag", "rag_hybrid", "system"]
            ]
            filtered_count = len(config["modules"])

            # 减少日志：仅在有过滤时输出
            if filtered_count < original_count:
                logger.info(
                    f"[{self.worker_id}] Gateway config filtered: "
                    f"{filtered_count}/{original_count} modules (RAG/RAG_HYBRID + system only)"
                )

        return config

    # NOTE: allocate_resource override removed to use unified batch allocation pattern
    # The parent class's default implementation now handles RAG resource allocation via:
    # 1. allocate_batch_resources(["rag"]) - allocates RAG resource from resource pool
    # 2. _sync_resource_sessions() - automatically syncs to RAG_SESSIONS
    # 3. setup_batch_resources() - calls rag_initialization() if configured
    # 4. get_batch_initial_observations() - gets initial state
    #
    # To provide RAG configuration (e.g., top_k), pass resource_init_data when calling
    # allocate_resource(), which will be forwarded to rag_initialization().

    # NOTE: release_resource and cleanup overrides also removed to use unified batch pattern
    # The parent class's default implementation handles resource release via:
    # 1. release_batch_resources() - releases all allocated resources
    # 2. _cleanup_resource_sessions() - clears local session caches
    # This ensures consistent resource lifecycle management across all resource types.