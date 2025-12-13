"""
MCP Server Adapter for Search V2 Services
Type: Standalone/Stateless Pattern
"""
import os
import json
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Annotated
from pydantic import Field
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
async def web_search(
    query: Annotated[str, Field(description="Search keywords or question text.", min_length=1)],
    k: Annotated[int, Field(description="Number of results to return; keep small to reduce noise.", ge=1, le=20)] = 5,
    region: Annotated[str, Field(description="Lowercase region/country code such as 'cn' or 'us'.", pattern="^[a-z]{2}$")] = "us",
) -> str:
    """
    Web search with summaries (stateless).
    Return shape (JSON string): List of items with
    - title, url (and link alias), snippet, source?, position?, date?
    - image_url/imageUrl?, thumbnail_url/thumbnailUrl?
    Format hints:
    - query: plain text (no URLs needed unless part of the question)
    - k: small integers (1-20) to control list length
    - region: two-letter lowercase region code (e.g., cn/us); defaults to cn
    """
    service = get_text_service()
    if not service:
        return "Error: TextSearchService not available (check dependencies/config)."

    try:
        results = await service.search_with_summaries(query=query, k=k, region=region)
        return json.dumps(results, ensure_ascii=False, indent=2)
    except Exception as e:
        import traceback
        print(f"[ERROR] web_search failed: {e}\n{traceback.format_exc()}")
        return json.dumps({"status": "error", "tool": "web_search", "message": str(e)}, ensure_ascii=False)

@ToolRegistry.register_tool("search_tools")
async def web_visit(
    url: Annotated[str, Field(description="Webpage URL (often taken from another search result).", min_length=1)],
    region: Annotated[str, Field(description="Lowercase region/country code such as 'cn' or 'us'.", pattern="^[a-z]{2}$")] = "us",
) -> str:
    """
    Visit a specific webpage by URL and return SerpAPI’s snippet/metadata for that page.
    """
    service = get_text_service()
    if not service:
        return "Error: TextSearchService not available (check dependencies/config)."
    try:
        text_summary = await service.visit_url(url=url, region=region)
        return json.dumps({"text": text_summary}, ensure_ascii=False, indent=2)
    except Exception as e:
        import traceback
        print(f"[ERROR] web_visit failed: {e}\n{traceback.format_exc()}")
        return json.dumps({"status": "error", "tool": "web_visit", "message": str(e)}, ensure_ascii=False)

@ToolRegistry.register_tool("search_tools")
async def image_search_by_text(
    query: Annotated[str, Field(description="Text query describing the target image (e.g., 'AIRFold logo').", min_length=1)],
    k: Annotated[int, Field(description="Number of image results to return.", ge=1, le=20)] = 5,
) -> str:
    """
    Search images by text and return URLs (thumbnail/full).
    Return shape (JSON string): List of items with
    - title, link
    - thumbnail/thumbnailUrl, image_url/imageUrl?, source?, domain?, position?
    Keep k small (1-20). Provide concise, visual-friendly keywords.
    """
    service = get_image_service()
    if not service:
        return "Error: ImageSearchService not available."

    try:
        results = await service.search_by_query(query=query, k=k)
        return json.dumps(results, ensure_ascii=False, indent=2)
    except Exception as e:
        import traceback
        print(f"[ERROR] image_search_by_text failed: {e}\n{traceback.format_exc()}")
        return json.dumps({"status": "error", "tool": "image_search_by_text", "message": str(e)}, ensure_ascii=False)

@ToolRegistry.register_tool("search_tools")
async def reverse_image_search(
    image_token: Annotated[str, Field(description="Image token from conversation history, e.g., 'image_1' for <image_1>.", min_length=1)],
    k: Annotated[int, Field(description="Number of matches to return.", ge=1, le=10)] = 3,
    messages: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Reverse image search using a referenced image token from the chat.
    Return shape (JSON string): List of items with
    - title, link
    - thumbnail/thumbnailUrl, image_url/imageUrl?, source?, domain?, position?
    Notes:
    - image_token must match a token already present in messages (<image_x> or <obs_x>).
    - messages is normally auto-injected by the client; callers usually omit it.
    """
    # First try to get the image processor to extract image source from token
    processor = get_image_processor()
    if not processor:
        return "Error: ImageProcessor not available."
        
    service = get_image_service()
    if not service:
        return "Error: ImageSearchService not available."

    try:
        # Extract image sources from messages
        image_sources = {}
        if messages:
            image_sources.update(processor.extract_sources_from_messages(messages))
        
        # Find the image URL corresponding to the token
        if image_token not in image_sources:
            return f"Error: Image token '{image_token}' not found in conversation history."
            
        image_detail = image_sources[image_token]
        image_url = image_detail.get("image_url", {}).get("url", "")
        if not image_url:
            return f"Error: Could not extract URL for image token '{image_token}'."
        
        # Perform reverse image search using the URL
        results = await service.search_by_image(image_input=image_url, k=k)
        return json.dumps(results, ensure_ascii=False, indent=2)
    except Exception as e:
        import traceback
        print(f"[ERROR] reverse_image_search failed: {e}\n{traceback.format_exc()}")
        return json.dumps({"status": "error", "tool": "reverse_image_search", "message": str(e)}, ensure_ascii=False)

@ToolRegistry.register_tool("search_tools",hidden=True)
async def upload_file_to_cloud(
    file_path: Annotated[str, Field(description="Absolute path to the local file to upload.", min_length=1)]
) -> str:
    """
    [Hidden] Upload a local file (e.g., screenshot) to cloud storage and return a public URL.
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
        import traceback
        print(f"[ERROR] upload_file_to_cloud failed: {e}\n{traceback.format_exc()}")
        return json.dumps({"status": "error", "tool": "upload_file_to_cloud", "message": str(e)}, ensure_ascii=False)

@ToolRegistry.register_tool("search_tools")
async def crop_images_by_token(
    crop_config: Annotated[
        Dict[str, List[int]],
        Field(description="Mapping of image tokens to crop boxes: {'<token>': [left, top, right, bottom]} (pixels).")
    ],
    messages: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Crop regions from images referenced by tokens in the conversation history.
    Return shape (JSON string):
    {
      "results": {"token": "<local path or url or error>", ...},
      "images": [<base64 or url>...]  # storage_mode=local -> base64; storage_mode=cloud -> url
    }
    Requirements:
    - Each crop box must be four integers: [left, top, right, bottom].
    - tokens must match existing <image_x> or <obs_x> entries.
    - messages is normally injected by the client.
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
        results: Dict[str, str] = await service.batch_crop_images(
            crop_config=formatted_config,
            messages=messages
        )
        images: List[str] = []
        storage_mode = getattr(service.config, "STORAGE_MODE", "cloud")

        for _, val in results.items():
            if not isinstance(val, str):
                continue
            # cloud: URL 直接返回
            if storage_mode == "cloud" and val.startswith(("http://", "https://")):
                images.append(val)
            # local: 读取文件为 base64
            elif storage_mode == "local":
                path = Path(val)
                if path.exists():
                    try:
                        with open(path, "rb") as f:
                            b64 = base64.b64encode(f.read()).decode("utf-8")
                            images.append(b64)
                    except Exception as _:
                        pass

        payload = {"results": results, "images": images}
        return json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception as e:
        import traceback
        print(f"[ERROR] crop_images_by_token failed: {e}\n{traceback.format_exc()}")
        return json.dumps({"status": "error", "tool": "crop_images_by_token", "message": str(e)}, ensure_ascii=False)
