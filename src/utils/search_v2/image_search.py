import asyncio
import aiohttp
import base64
import uuid
import time
from pathlib import Path
from typing import List, Dict, Union
from urllib.parse import quote

from .config.settings import Config
from .cloud_storage import CloudStorageService


class ImageSearchService:
    """Service for handling image search functionality"""
    
    def __init__(self):
        self.config = Config()
        self.cloud_storage = CloudStorageService()
    
    async def search_by_image(self, image_input: Union[str, Path], k: int = None) -> List[Dict[str, str]]:
        """
        Search for similar images using reverse image search
        
        Args:
            image_input: Either a URL string or a Path to a local image file
            k: Number of results to return (default from config)
            
        Returns:
            List of dictionaries containing title, thumbnail, and link
        """
        if k is None:
            k = self.config.DEFAULT_IMAGE_RESULTS
        
        # Validate API key
        if not self.config.SERPAPI_API_KEY:
            raise ValueError("SERPAPI_API_KEY is required for image search")
        
        # Determine if input is URL or local file and get search URL
        image_url = await self._prepare_image_url(image_input)
        
        # Perform the actual search
        async with aiohttp.ClientSession() as session:
            return await self._search_by_url(session, image_url, k)
    
    async def _prepare_image_url(self, image_input: Union[str, Path]) -> str:
        """
        Prepare image URL for search - handle URL, Base64, or local files
        """
        # =================================================================
        # 1. Handle String Input (URL or Base64)
        # =================================================================
        if isinstance(image_input, str):
            # --- Case A: Base64 String ---
            if image_input.startswith("data:image"):
                try:
                    # 1. 解析 Base64 头，提取扩展名
                    if "," in image_input:
                        header, encoded = image_input.split(",", 1)
                        # header 格式如: data:image/png;base64
                        file_ext = header.split(";")[0].split("/")[1]
                    else:
                        raise ValueError("Invalid Base64 format: missing header")

                    # 2. 使用专用的 cache 目录
                    filename = f"b64_search_{int(time.time())}_{uuid.uuid4().hex[:6]}.{file_ext}"
                    cache_path = self.config.SEARCH_CACHE_DIR / filename

                    # 3. 解码并保存到缓存文件
                    with open(cache_path, "wb") as f:
                        f.write(base64.b64decode(encoded))

                    try:
                        # 4. 上传到云端
                        print(f"[INFO] Auto-uploading Base64 image: {filename}")
                        upload_result = await self.cloud_storage.upload_single_image(cache_path)
                        return upload_result['url']
                    finally:
                        # 5. 缓存清理控制
                        if not self.config.KEEP_LOCAL_CACHE:
                            if cache_path.exists():
                                cache_path.unlink()
                        else:
                            print(f"[DEBUG] Kept Base64 cache: {filename}")

                except Exception as e:
                    raise ValueError(f"Base64 processing failed: {e}")

            # --- Case B: HTTP/HTTPS URL ---
            elif image_input.startswith(('http://', 'https://')):
                return image_input

            # --- Case C: Local File Path (String format) ---
            else:
                image_input = Path(image_input)

        # =================================================================
        # 2. Handle Path Input (Local File)
        # =================================================================
        if isinstance(image_input, Path):
            if not image_input.exists():
                raise FileNotFoundError(f"File not found: {image_input}")

            if not self.cloud_storage.is_supported_image(image_input):
                raise ValueError(f"Unsupported image format: {image_input.suffix}")

            # 本地文件也是自动上传，转为 URL 供 SerpAPI 使用
            print(f"[INFO] Auto-uploading local file: {image_input.name}")
            upload_result = await self.cloud_storage.upload_single_image(image_input)
            return upload_result['url']

        raise ValueError(f"Invalid image input type: {type(image_input)}")
    
    async def _search_by_url(self, session: aiohttp.ClientSession, image_url: str, k: int) -> List[Dict[str, str]]:
        """
        Perform reverse image search using SerpAPI with multiple engines
        """
        # Define search engines in order of preference
        engines = [
            ("google_lens", {"url": image_url}, "visual_matches"),
            ("google_reverse_image", {"image_url": image_url}, "image_results"),
            ("google_images", {"q": image_url}, "images_results"),
        ]
        
        results = []
        
        for engine_name, extra_params, field in engines:
            print(f"[INFO] Trying {engine_name} for image search...")
            
            params = {
                "engine": engine_name,
                "api_key": self.config.SERPAPI_API_KEY
            }
            params.update(extra_params)
            
            try:
                async with session.get(
                    self.config.SERPAPI_URL, 
                    params=params, 
                    timeout=self.config.REQUEST_TIMEOUT
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        print(f"[WARN] {engine_name} HTTP {resp.status}: {error_text}")
                        continue
                    
                    data = await resp.json()
                    
            except Exception as e:
                print(f"[WARN] {engine_name} request failed: {e}")
                continue
            
            # Extract results from response
            items = data.get(field, [])
            for item in items:
                entry = self._extract_search_result(item)
                if entry and entry["link"] and entry["thumbnail"]:
                    results.append(entry)
                
                if len(results) >= k:
                    break
            
            # If we got results from this engine, stop trying others
            if results:
                print(f"[INFO] Found {len(results)} results using {engine_name}")
                break
        
        # Remove duplicates based on link
        return self._deduplicate_results(results)[:k]
    
    def _extract_search_result(self, item: dict) -> Dict[str, str]:
        """Extract and normalize search result from API response"""
        return {
            "title": item.get("title") or item.get("source") or "No Title",
            "thumbnail": (
                item.get("thumbnail") or 
                item.get("thumbnail_url") or 
                item.get("original") or 
                ""
            ),
            "link": (
                item.get("link") or 
                item.get("source") or 
                item.get("image") or 
                ""
            ),
        }
    
    def _deduplicate_results(self, results: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Remove duplicate results based on link"""
        seen_links = set()
        deduplicated = []
        
        for result in results:
            link = result.get("link", "")
            if link and link not in seen_links:
                seen_links.add(link)
                deduplicated.append(result)
        
        return deduplicated
    
    async def batch_search_images(self, image_inputs: List[Union[str, Path]], 
                                k: int = None) -> List[List[Dict[str, str]]]:
        """Search multiple images concurrently"""
        if k is None:
            k = self.config.DEFAULT_IMAGE_RESULTS
        
        tasks = [self.search_by_image(image_input, k) for image_input in image_inputs]
        return await asyncio.gather(*tasks, return_exceptions=True)