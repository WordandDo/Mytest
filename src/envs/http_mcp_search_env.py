# src/envs/http_mcp_search_env.py
import logging
from typing import Dict, Any, Optional, List
from .http_mcp_env import HttpMCPEnv

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_SEARCH = """You are a capable Search & Analysis Agent. 
Your goal is to gather information, analyze content, and process visual data using the provided tools.

## Capabilities
1. **Web Search**: Use `web_search` to find up-to-date information, news, and technical documentation.
2. **Visual Analysis**: 
   - Use `reverse_image_search` to find the source or similar images of a given URL.
   - Use `crop_images_by_token` to focus on specific regions of an image referenced in the conversation history.

## Tool Usage Strategy
1. **Search Broadly, Then Narrow Down**: Start with broad keywords, then refine based on initial results.
2. **Cross-Verify**: If a search result is ambiguous, verify it with a second source or image search.
3. **Visual Context**: When handling images, pay attention to the specific tokens (e.g., <img_1>) provided in the context.
4. **Efficiency**: Do not make redundant queries. Read the search summaries carefully.

## Answer Strategy
- Provide comprehensive answers based strictly on the tool outputs.
- Cite your sources (URLs) when providing facts.
- If the final answer involves an image, ensure you have processed it correctly.
"""

class HttpMCPSearchEnv(HttpMCPEnv):
    """
    Search-focused environment that inherits from HttpMCPEnv.
    
    This environment is specialized for 'utility' type resources (like search_v2),
    which are typically stateless and do not require heavy resource allocation (locking).
    """

    def __init__(self,
                 model_name: str = "gpt-4.1-2025-04-14",
                 parallel_degree: int = 1,
                 **kwargs):

        # 确保使用默认网关配置路径，除非外部覆盖
        if "gateway_config_path" not in kwargs:
            kwargs["gateway_config_path"] = "gateway_config.json"

        # 初始化父类
        super().__init__(
            model_name=model_name,
            parallel_degree=parallel_degree,
            **kwargs
        )

        # [关键兼容性设置]
        # Search V2 (utility) 是无状态服务，不需要向 Resource Manager 申请锁定/分配。
        # 父类 HttpMCPEnv 默认会把所有非 system 的 module 加入 active_resources 并尝试 allocate。
        # 这里我们需要清空 active_resources，避免 allocate_batch_resources 报错或做无用功。
        self.active_resources = []
        
        logger.info(f"HttpMCPSearchEnv initialized for {self.worker_id} (Stateless Mode)")

    @property
    def mode(self) -> str:
        """定义新的环境模式名称"""
        return "http_mcp_search"

    def get_system_prompt(self, task_question: Optional[str] = None, **kwargs) -> str:
        """
        重写 System Prompt，注入搜索专用的提示词和工具描述
        """
        prompt = SYSTEM_PROMPT_SEARCH

        # 动态注入工具描述 (由父类从 MCP Server 获取)
        tool_descriptions = self.get_tool_descriptions()
        if tool_descriptions:
            prompt += f"\n\n## Available Tools\n{tool_descriptions}"

        # 注入当前任务
        if task_question:
            prompt += f"\n\n## Current Task\n{task_question}"

        return prompt

    def _load_gateway_config(self, config_path: str) -> Dict[str, Any]:
        """
        重写配置加载逻辑：只加载 'utility' 类型的模块 (对应 search_tools)
        """
        config = super()._load_gateway_config(config_path)

        if "modules" in config:
            original_count = len(config["modules"])
            # 过滤只保留 utility 类型的模块 (我们在 server 端将 search 定义为了 utility)
            config["modules"] = [
                module for module in config["modules"]
                if module.get("resource_type") == "utility"
            ]
            
            # 如果没有找到 utility，尝试找包含 search_tools 的模块作为回退
            if not config["modules"]:
                config["modules"] = [
                    module for module in super()._load_gateway_config(config_path).get("modules", [])
                    if any("search" in g for g in module.get("tool_groups", []))
                ]

            filtered_count = len(config["modules"])
            if filtered_count < original_count:
                logger.info(
                    f"[{self.worker_id}] Gateway config filtered: "
                    f"{filtered_count}/{original_count} modules (Search/Utility only)"
                )

        return config