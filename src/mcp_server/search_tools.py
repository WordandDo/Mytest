"""
MCP Tools for Search Services
"""

import sys
import os
from typing import List, Dict, Any, Optional

# Add the src directory to the Python path for absolute imports
src_path = os.path.join(os.path.dirname(__file__), '..')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Try to import the search services
try:
    # Use absolute import from utils.search_v2 package
    from utils.search_v2 import TextSearchService, ImageSearchService
    TEXT_SEARCH_AVAILABLE = True
    IMAGE_SEARCH_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import search services: {e}")
    TEXT_SEARCH_AVAILABLE = False
    IMAGE_SEARCH_AVAILABLE = False

# Import the ToolRegistry
try:
    # Add the mcp_server directory to the Python path for relative imports
    mcp_server_path = os.path.dirname(__file__)
    if mcp_server_path not in sys.path:
        sys.path.insert(0, mcp_server_path)
    
    from core.tool import ToolRegistry
    MCP_CORE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import MCP core: {e}")
    MCP_CORE_AVAILABLE = False


def _ensure_event_loop():
    """Ensure an event loop is available for asyncio operations"""
    import asyncio
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


if MCP_CORE_AVAILABLE and (TEXT_SEARCH_AVAILABLE or IMAGE_SEARCH_AVAILABLE):
    @ToolRegistry.register_tool("search_text")
    def search_text(
        query: str,
        num_results: Optional[int] = None,
        region: Optional[str] = None,
        lang: Optional[str] = None,
        llm_model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Performs a text search using the TextSearchService.

        Args:
            query: The search query string
            num_results: Number of results to return
            region: Search region (e.g., 'us', 'cn')
            lang: Search language (e.g., 'en', 'zh-CN')
            llm_model: LLM model to use for summaries

        Returns:
            A dictionary containing search results
        """
        if not TEXT_SEARCH_AVAILABLE:
            return {"error": "Text search service not available"}
            
        try:
            _ensure_event_loop()
            import asyncio
            service = TextSearchService()
            results = asyncio.run(service.search_with_summaries(
                query=query,
                k=num_results,
                region=region,
                lang=lang,
                llm_model=llm_model
            ))
            return {"results": results, "status": "success"}
        except Exception as e:
            return {"error": f"Search failed: {str(e)}", "status": "error"}


    @ToolRegistry.register_tool("search_images")
    def search_images(
        query: str,
        num_results: Optional[int] = None,
        image_size: str = "medium"
    ) -> Dict[str, Any]:
        """
        Performs an image search using the ImageSearchService.

        Args:
            query: The search query string
            num_results: Number of results to return
            image_size: Size of images to return - small, medium, large (default: medium)

        Returns:
            A dictionary containing image search results
        """
        if not IMAGE_SEARCH_AVAILABLE:
            return {"error": "Image search service not available"}
            
        try:
            _ensure_event_loop()
            import asyncio
            service = ImageSearchService()
            results = asyncio.run(service.search(query, k=num_results, size=image_size))
            return {"results": results, "status": "success"}
        except Exception as e:
            return {"error": f"Image search failed: {str(e)}", "status": "error"}


    @ToolRegistry.register_tool("search_images_by_image")
    def search_images_by_image(
        image_input: str,
        num_results: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Performs a reverse image search using the ImageSearchService.

        Args:
            image_input: Either a URL string or a Path to a local image file
            num_results: Number of results to return

        Returns:
            A dictionary containing image search results
        """
        if not IMAGE_SEARCH_AVAILABLE:
            return {"error": "Image search service not available"}
            
        try:
            _ensure_event_loop()
            import asyncio
            service = ImageSearchService()
            results = asyncio.run(service.search_by_image(image_input, k=num_results))
            return {"results": results, "status": "success"}
        except Exception as e:
            return {"error": f"Reverse image search failed: {str(e)}", "status": "error"}

else:
    # Create dummy functions if MCP core is not available
    def search_text(*args, **kwargs):
        return {"error": "MCP core or search services not available"}
    
    def search_images(*args, **kwargs):
        return {"error": "MCP core or search services not available"}
    
    def search_images_by_image(*args, **kwargs):
        return {"error": "MCP core or search services not available"}