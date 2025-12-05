# src/envs/http_mcp_rag_env.py
import logging
from typing import Dict, Any, Optional
from .http_mcp_env import HttpMCPEnv

logger = logging.getLogger(__name__)

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
The final answer only contains the short answer to the question (few words), no other words like reasoning content.
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

        # Initialize parent class
        super().__init__(
            model_name=model_name,
            parallel_degree=parallel_degree,
            **kwargs
        )

        logger.info(f"HttpMCPRagEnv initialized for {self.worker_id}")

    @property
    def mode(self) -> str:
        """Override mode to identify this as RAG-specific environment"""
        return "http_mcp_rag"

    def get_system_prompt(self, task_question: Optional[str] = None, **kwargs) -> str:
        """
        Override to use RAG-specific system prompt

        Args:
            task_question: The current task question (optional)
            **kwargs: Additional keyword arguments

        Returns:
            The customized system prompt for RAG tasks
        """
        # Start with the RAG-specific prompt
        prompt = SYSTEM_PROMPT_GENERIC

        # Add tool descriptions
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
        """
        config = super()._load_gateway_config(config_path)

        # Filter modules to only include RAG resources
        if "modules" in config:
            original_count = len(config["modules"])
            config["modules"] = [
                module for module in config["modules"]
                if module.get("resource_type") == "rag"
            ]
            filtered_count = len(config["modules"])

            if filtered_count < original_count:
                logger.info(
                    f"[{self.worker_id}] Filtered gateway config: "
                    f"{filtered_count}/{original_count} modules (RAG only)"
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

