"""
MCP Server Adapter for Search V2 Services
Type: Standalone/Stateless Pattern
"""
import os
import json
from typing import List, Dict, Any, Optional, Tuple
from mcp_server.core.registry import ToolRegistry

# 尝试导入 search_v2 服务
try:
    from src.utils.search_v2 import (
        TextSearchService, 
        ImageSearchService, 
        CloudStorageService,
        ImageProcessor
    )
    SEARCH_V2_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Search V2 services not found: {e}")
    SEARCH_V2_AVAILABLE = False

# ==============================================================================
# Service Initialization (Singleton)
# ==============================================================================
# 这些服务会在模块加载时初始化，它们内部会读取 .env 中的配置
_text_service = None
_image_service = None
_cloud_service = None
_image_processor = None

def get_text_service():
    global _text_service
    if not _text_service and SEARCH_V2_AVAILABLE:
        _text_service = TextSearchService()
    return _text_service

def get_image_service():
    global _image_service
    if not _image_service and SEARCH_V2_AVAILABLE:
        _image_service = ImageSearchService()
    return _image_service

def get_cloud_service():
    global _cloud_service
    if not _cloud_service and SEARCH_V2_AVAILABLE:
        _cloud_service = CloudStorageService()
    return _cloud_service

def get_image_processor():
    global _image_processor
    if not _image_processor and SEARCH_V2_AVAILABLE:
        _image_processor = ImageProcessor()
    return _image_processor

# ==============================================================================
# Tool Definitions
# Group: search_tools
# ==============================================================================

@ToolRegistry.register_tool("search_tools")
async def web_search(query: str, k: int = 5, region: str = "cn") -> str:
    """
    Perform a web search to get relevant information and summaries.
    
    Args:
        query: The search query string.
        k: Number of results to return (default 5).
        region: Search region code (e.g., 'cn', 'us').
    """
    service = get_text_service()
    if not service:
        return "Error: TextSearchService not available (check dependencies/config)."

    try:
        results = await service.search_with_summaries(query=query, k=k, region=region)
        return json.dumps(results, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error executing web search: {str(e)}"

@ToolRegistry.register_tool("search_tools")
async def reverse_image_search(image_url: str, k: int = 3) -> str:
    """
    Search for information about an image using its URL (Google Lens/Reverse Search).
    
    Args:
        image_url: The URL of the image to search for.
        k: Number of results to return.
    """
    service = get_image_service()
    if not service:
        return "Error: ImageSearchService not available."

    try:
        results = await service.search_by_image(image_input=image_url, k=k)
        return json.dumps(results, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error executing image search: {str(e)}"

@ToolRegistry.register_tool("search_tools", hidden=True)
async def upload_file_to_cloud(file_path: str) -> str:
    """
    Upload a local file (e.g., a screenshot taken by the agent) to cloud storage and get a public URL.
    
    Args:
        file_path: The absolute path to the local file.
    """
    service = get_cloud_service()
    if not service:
        return "Error: CloudStorageService not available."

    try:
        from pathlib import Path
        path_obj = Path(file_path)
        if not path_obj.exists():
            return f"Error: File not found at {file_path}"

        result = await service.upload_single_image(path_obj)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error uploading file: {str(e)}"

@ToolRegistry.register_tool("search_tools")
async def crop_images_by_token(
    crop_config: Dict[str, List[int]],
    messages: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Crop specific regions from images in the conversation history based on tokens.

    Args:
        crop_config: Dictionary mapping tokens to crop boxes. Format: {"<token>": [left, top, right, bottom]}.
        messages: (Optional) The conversation history containing images. usually injected by the system, user doesn't need to provide.
    """
    service = get_image_processor()
    if not service:
        return "Error: ImageProcessor not available."

    try:
        # 转换 crop_config 的 List 为 Tuple，适配 PIL
        formatted_config = {}
        for k, v in crop_config.items():
            if len(v) != 4:
                return f"Error: Crop box for {k} must have 4 integers [left, top, right, bottom]"
            formatted_config[k] = tuple(v)

        # 注意：messages 参数依赖于 Gateway/Client 端的注入机制
        # 如果未注入，工具会尝试仅处理 content (如果被错误传入) 或报错
        results = await service.batch_crop_images(
            crop_config=formatted_config,
            messages=messages
        )
        return json.dumps(results, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error cropping images: {str(e)}"