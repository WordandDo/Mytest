import io
import base64
import requests
import re
import time
import uuid
from PIL import Image
from typing import List, Dict, Any, Tuple, Optional, Union
from pathlib import Path

from .config.settings import Config
from .cloud_storage import CloudStorageService

class ImageProcessor:
    """
    处理 OpenAI 消息中的图片，支持基于 Token 的定位和裁切。
    
    【内存优化版】
    - 解析阶段仅提取元数据，不加载图片实体。
    - 裁切阶段按需加载，用完即毁，最小化内存占用。
    """
    
    # 添加裁切产物的特征标记
    CROP_MARKER = "mcp_derived_crop_"
    
    def __init__(self):
        self.config = Config()
        self.cloud_storage = CloudStorageService()
        self.config.create_directories()

    def _load_image_from_source(self, image_detail: Dict[str, Any]) -> Optional[Image.Image]:
        """
        [即时加载] 从 image_url 结构中提取并解码图片。
        只有在真正需要处理图片时才调用此方法。
        """
        url_str = image_detail.get("image_url", {}).get("url", "")
        
        if not url_str:
            return None

        try:
            # === 模式 A: Base64 图片 ===
            if url_str.startswith("data:image"):
                if "," in url_str:
                    header, encoded = url_str.split(",", 1)
                else:
                    encoded = url_str
                
                # 解码 Base64 -> Bytes -> Image
                return Image.open(io.BytesIO(base64.b64decode(encoded)))
            
            # === 模式 B: HTTP/HTTPS URL ===
            elif url_str.startswith(("http://", "https://")):
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
                # 使用 stream=True 可以稍微优化大文件的网络IO，但PIL需要完整文件流
                resp = requests.get(url_str, headers=headers, timeout=self.config.REQUEST_TIMEOUT)
                resp.raise_for_status()
                return Image.open(io.BytesIO(resp.content))
            
            else:
                print(f"[WARN] Unknown image format prefix: {url_str[:30]}...")
                return None

        except Exception as e:
            print(f"[ERROR] Failed to load image data: {e}")
            return None

    def _map_tokens_to_sources(self, content: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        [轻量级解析] 扫描 content，建立 Token 到 图片元数据 的映射。
        注意：此处不下载/解码图片，只保存引用。
        """
        source_map = {}
        last_token = None
        
        token_pattern = re.compile(r"<([a-zA-Z0-9_]+)>\s*$")

        for item in content:
            item_type = item.get("type")
            
            if item_type == "text":
                text = item.get("text", "")
                match = token_pattern.search(text)
                if match:
                    last_token = match.group(1)
                else:
                    if text.strip():
                        last_token = None
            
            elif item_type == "image_url":
                if last_token:
                    # 关键修改：只存储 item (元数据)，不调用 _load_image_from_source
                    source_map[last_token] = item
                    # 消费 Token
                    last_token = None

        return source_map

    def extract_sources_from_messages(self, messages: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        遍历对话历史，提取所有 Token 对应的图片元数据。
        """
        all_sources = {}
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                sources = self._map_tokens_to_sources(content)
                all_sources.update(sources)
        return all_sources

    async def batch_crop_images(self,
                              crop_config: Dict[str, Tuple[int, int, int, int]],
                              messages: Optional[List[Dict[str, Any]]] = None,
                              content: Optional[List[Dict[str, Any]]] = None) -> Dict[str, str]:
        """
        执行批量裁切任务（内存优化版）。
        存储模式由配置文件的 STORAGE_MODE 决定，不再通过参数传递。
        """
        # 1. 建立索引 (内存占用极小)
        # 仅存储 Token 字符串和 URL 字符串
        image_sources = {}
        if content:
            image_sources.update(self._map_tokens_to_sources(content))
        if messages:
            image_sources.update(self.extract_sources_from_messages(messages))
            
        if not image_sources:
            print("[WARN] No images found in the provided context.")
            return {k: "Error: No images found for this token" for k in crop_config.keys()}
        
        results = {}
        # 使用专用的裁切目录
        local_crop_dir = self.config.CROPS_DIR
        local_crop_dir.mkdir(parents=True, exist_ok=True)

        # 读取全局存储模式配置
        current_mode = self.config.STORAGE_MODE
        
        # 2. 串行处理每个裁切任务 (保证同一时间内存中只有一张图片)
        for token, box in crop_config.items():
            if token not in image_sources:
                print(f"[WARN] Token <{token}> not found in extracted images.")
                results[token] = f"Error: Image <{token}> not found"
                continue
            
            # === 关键：按需加载 ===
            source_item = image_sources[token]
            
            # 拦截逻辑：禁止二次裁切（通过文件名标记与 URL 子串同时判断）
            url_str = source_item.get("image_url", {}).get("url", "")
            if self.CROP_MARKER in url_str:
                msg = f"Error: recursive cropping is not allowed. Image <{token}> is already a cropped fragment."
                print(f"[INFO] Blocked recursive crop for <{token}>")
                results[token] = msg
                continue
            
            img = None
            try:
                # 此时才真正占用内存加载图片
                img = self._load_image_from_source(source_item)
                
                if not img:
                    raise ValueError("Failed to load image content")

                # 验证裁切框
                if len(box) != 4:
                    raise ValueError(f"Invalid crop box format: {box}")
                
                # 执行裁切
                cropped_img = img.crop(box)
                
                # 保存
                fmt = img.format if img.format else 'PNG'
                # 强制添加前缀
                filename = f"{self.CROP_MARKER}{token}_{int(time.time())}_{uuid.uuid4().hex[:6]}.{fmt.lower()}"
                file_path = local_crop_dir / filename
                
                cropped_img.save(file_path, format=fmt)

                # 根据配置决定存储方式
                if current_mode == "local":
                    results[token] = str(file_path.absolute())

                elif current_mode == "cloud":
                    try:
                        upload_res = await self.cloud_storage.upload_single_image(file_path)
                        results[token] = upload_res['url']

                        # 缓存清理控制
                        if not self.config.KEEP_LOCAL_CACHE:
                            file_path.unlink()
                        else:
                            print(f"[DEBUG] Kept local crop cache: {file_path.name}")

                    except Exception as upload_err:
                        results[token] = f"Error upload: {upload_err}"

                else:
                    results[token] = f"Error: Unknown STORAGE_MODE: {current_mode}"

            except Exception as e:
                print(f"[ERROR] Failed to crop <{token}>: {e}")
                results[token] = f"Error: {str(e)}"
            
            finally:
                # === 关键：显式释放内存 ===
                if img:
                    img.close()
                    del img
                # 提示垃圾回收（在极高并发下可选，一般 Python 会自动处理）
                # import gc; gc.collect()
                
        return results
