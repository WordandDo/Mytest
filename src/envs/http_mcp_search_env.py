# src/envs/http_mcp_search_env.py
import logging
from typing import Dict, Any, Optional, List
from .http_mcp_env import HttpMCPEnv
import os
import base64

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_SEARCH = """You are a capable Search & Analysis Agent. 
Your goal is to gather information, analyze content, and process visual data using the provided tools.

## Capabilities
1. **Web Search**: Use `web_search` to find up-to-date information, news, and technical documentation.
2. **Visual Analysis**: 
   - Use `reverse_image_search` to find the source or similar images of a given URL.
   - Use `crop_images_by_token` to focus on specific regions of an image referenced in the conversation history.
3. **Image by Text**:
   - Use `image_search_by_text` to find images related to a text query.

## Tool Usage Strategy
1. **Search Broadly, Then Narrow Down**: Start with broad keywords, then refine based on initial results.
2. **Cross-Verify**: If a search result is ambiguous, verify it with a second source or image search.
3. **Visual Context**: When handling images, pay attention to the specific tokens (e.g., <image_1>) provided in the context.
4. **Efficiency**: Do not make redundant queries. Read the search summaries carefully.

## Answer Strategy
- Provide comprehensive answers based strictly on the tool outputs.
- Cite your sources (URLs) when providing facts.
- If the final answer involves an image, ensure you have processed it correctly.

## Image Token Policy
- Input images from the task are wrapped as paired tokens: <image_1> ... </image_1>, <image_2> ... </image_2>, etc.
- Images produced by tools (e.g., search results thumbnails) are wrapped in order as <obs_i> ... </obs_i> and can be cropped by referencing obs_i.
- Cropped images must NOT be cropped again; they are not re-injected with tokens.
- Prefer reusing already tokenized images (<image_k> / <obs_i>) for reverse-image or cropping steps.
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

        # [新增] 默认白名单：只暴露 Search 相关工具给 Agent
        # 若外部未显式传入 `tool_whitelist`，则启用以下默认集合
        if not kwargs.get("tool_whitelist"):
            kwargs["tool_whitelist"] = [
                "web_search",
                "reverse_image_search",
                "crop_images_by_token",
                "image_search_by_text",
            ]

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

    def run_task(self, task: Dict[str, Any], agent_config: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
        """
        扩展：支持问题相关图片输入。
        - 从 task.metadata.images 读取图片（本地路径或 URL）
        - 本地文件存在则转为 base64；URL 直接注入
        - 基类会在用户消息中按 <img_n> 注入这些内容
        """
        # 清理之前的注入
        self.input_images = []

        try:
            md = task.get("metadata", {}) if isinstance(task.get("metadata"), dict) else {}
            imgs = md.get("images")
            images: list[str] = []
            if isinstance(imgs, str):
                images = [imgs]
            elif isinstance(imgs, list):
                images = [x for x in imgs if isinstance(x, str)]

            for p in images:
                s = p.strip()
                if s.startswith("http://") or s.startswith("https://"):
                    self.input_images.append({"url": s})
                    continue
                if os.path.exists(s):
                    try:
                        with open(s, 'rb') as f:
                            b64 = base64.b64encode(f.read()).decode('utf-8')
                            self.input_images.append({"b64": b64})
                    except Exception:
                        # 忽略单个文件失败
                        pass
        except Exception:
            self.input_images = []

        return super().run_task(task, agent_config, logger)
