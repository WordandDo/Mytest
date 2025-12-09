import asyncio
import time
import json
import csv
import hashlib
import math
import requests
from pathlib import Path
from typing import Dict, List, Optional, Callable

from .config.settings import Config


class CloudStorageService:
    """
    Service for handling file uploads to 123Pan cloud storage via Official Open API.
    Implemented directly using requests to bypass SDK limitations.
    """
    
    # 123云盘开放平台官方 API 地址
    API_BASE_URL = "https://open-api.123pan.com"
    
    def __init__(self):
        self.config = Config()
        self._access_token = self._get_access_token()
        self._headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
            "Platform": "open_platform"
        }
    
    def _get_access_token(self) -> str:
        """Get access token directly from configuration"""
        token = self.config.PAN123_ACCESS_TOKEN
        if not token:
            raise ValueError(
                "PAN123_ACCESS_TOKEN is not set. Please set this environment variable "
                "with your 123Pan access token string."
            )
        return token

    def _calculate_md5(self, file_path: Path) -> str:
        """Calculate file MD5 hash"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _api_request(self, method: str, endpoint: str, data: dict = None, max_retries: int = 3) -> dict:
        """Helper to make API requests with error handling"""
        url = f"{self.API_BASE_URL}{endpoint}"
        
        for attempt in range(max_retries):
            try:
                if method.upper() == "GET":
                    response = requests.get(url, params=data, headers=self._headers, timeout=30)
                elif method.upper() == "PUT":
                    # PUT通常用于上传二进制流，这里做特殊处理，data作为body
                    response = requests.put(url, data=data, timeout=60)
                    if response.status_code == 200:
                        return {} # PUT通常只看状态码
                    response.raise_for_status()
                else:
                    response = requests.post(url, json=data, headers=self._headers, timeout=30)
                
                # 解析响应
                if method.upper() != "PUT":
                    res_json = response.json()
                    if res_json.get("code") != 0:
                        raise ValueError(f"API Error ({res_json.get('code')}): {res_json.get('message')}")
                    return res_json.get("data", {})
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(1)
        return {}

    def _upload_logic_sync(self, file_path: Path, parent_id: int) -> Dict[str, str]:
        """
        Synchronous implementation of the upload logic chain:
        V2 Create -> V1 Upload -> V1 Complete (Polling) -> Get Direct Link
        """
        file_size = file_path.stat().st_size
        file_name = file_path.name
        file_md5 = self._calculate_md5(file_path)
        
        # --- Step 1: Create Task (使用 V2 接口) ---
        create_data = {
            "parentFileID": parent_id,
            "filename": file_name,
            "etag": file_md5,
            "size": file_size,
            "duplicate": 1  # 1=自动重命名
        }
        # 注意：这里必须用 V2
        file_meta = self._api_request("POST", "/upload/v2/file/create", create_data)
        
        # 处理秒传
        if file_meta.get("reuse"):
            file_id = file_meta.get("fileID")
            # 秒传直接获取直链
            return self._get_direct_link_result(file_name, file_id)

        # --- Step 2: Upload Chunks (使用 V1 接口获取地址) ---
        preupload_id = file_meta["preuploadID"]
        slice_size = file_meta["sliceSize"]
        num_slices = math.ceil(file_size / slice_size)
        
        with open(file_path, "rb") as f:
            for i in range(1, num_slices + 1):
                # 1. 获取上传 URL (必须用 V1)
                get_url_data = {"preuploadID": preupload_id, "sliceNo": i}
                url_resp = self._api_request("POST", "/upload/v1/file/get_upload_url", get_url_data)
                upload_url = url_resp["presignedURL"]
                
                # 2. 上传数据
                chunk = f.read(slice_size)
                # PUT请求不使用通用Header，直接发二进制
                requests.put(upload_url, data=chunk).raise_for_status()

        # --- Step 3: Complete Upload (使用 V1 接口) ---
        complete_data = {"preuploadID": preupload_id}
        result = self._api_request("POST", "/upload/v1/file/upload_complete", complete_data)
        
        # --- Step 4: Handle Async Processing (轮询) ---
        if result.get("async") and not result.get("completed"):
            print(f"[INFO] Server processing asynchronously for {file_name}...")
            file_id = None
            # 轮询最多 60秒
            for _ in range(30):
                time.sleep(2)
                async_res = self._api_request("POST", "/upload/v1/file/upload_async_result", complete_data)
                if async_res.get("completed"):
                    file_id = async_res.get("fileID")
                    break
            
            if not file_id:
                raise RuntimeError("Upload processing timed out (Async result not completed)")
        else:
            file_id = result.get("fileID")

        if not file_id:
            raise RuntimeError(f"Failed to get fileID after upload. Result: {result}")

        # --- Step 5: Get Direct Link ---
        return self._get_direct_link_result(file_name, file_id)

    def _get_direct_link_result(self, file_name: str, file_id: str) -> Dict[str, str]:
        """Call direct-link API to get the download URL"""
        # 兼容 fileID 为 int 或 str
        link_res = self._api_request("GET", "/api/v1/direct-link/url", {"fileID": file_id})
        url = link_res.get("url")
        
        if not url:
            raise ValueError(f"Failed to retrieve direct link for fileID: {file_id}")
            
        return {
            "name": file_name,
            "fileID": str(file_id),
            "url": url
        }

    async def upload_single_image(self, file_path: Path) -> Dict[str, str]:
        """
        Upload a single image file to cloud storage (Async Wrapper)
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() not in self.config.SUPPORTED_IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        last_error = None
        
        # Retry logic for the entire process
        for attempt in range(1, self.config.MAX_RETRIES + 1):
            try:
                # Run blocking sync upload logic in a thread
                result = await asyncio.to_thread(
                    self._upload_logic_sync,
                    file_path,
                    int(self.config.PAN123_PARENT_FILE_ID)
                )
                return result
                
            except Exception as e:
                last_error = e
                print(f"[WARN] Upload failed ({file_path.name}), attempt {attempt}/{self.config.MAX_RETRIES}: {e}")
                
                if attempt < self.config.MAX_RETRIES:
                    await asyncio.sleep(self.config.RETRY_SLEEP)
        
        raise RuntimeError(f"Upload permanently failed for {file_path} — last error: {last_error}")
    
    async def upload_multiple_images(self, file_paths: List[Path], 
                                   progress_callback: Optional[Callable] = None) -> List[Dict[str, str]]:
        """
        Upload multiple image files to cloud storage
        """
        results = []
        total_files = len(file_paths)
        
        for idx, file_path in enumerate(file_paths, 1):
            if progress_callback:
                progress_callback(idx, total_files, file_path.name)
            
            try:
                result = await self.upload_single_image(file_path)
                results.append(result)
                print(f"[OK] {file_path.name} -> {result['url']}")
            except Exception as e:
                print(f"[ERROR] Failed to upload {file_path.name}: {e}")
                results.append({
                    "name": file_path.name,
                    "fileID": "",
                    "url": "",
                    "error": str(e)
                })
        
        return results
    
    def save_upload_mapping(self, results: List[Dict[str, str]], 
                           csv_path: Optional[Path] = None,
                           json_path: Optional[Path] = None) -> None:
        """
        Save upload results to CSV and JSON files
        """
        if not csv_path:
            csv_path = self.config.LOGS_DIR / "upload_mapping.csv"
        if not json_path:
            json_path = self.config.LOGS_DIR / "upload_mapping.json"
        
        # Ensure log directory exists
        self.config.create_directories()
        
        # Save to CSV
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ["name", "fileID", "url"]
            if any("error" in result for result in results):
                fieldnames.append("error")
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        # Save to JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"[INFO] Upload mapping saved to {csv_path} and {json_path}")
    
    def is_supported_image(self, file_path: Path) -> bool:
        """Check if file is a supported image type"""
        return file_path.suffix.lower() in self.config.SUPPORTED_IMAGE_EXTENSIONS
    
    def find_images_in_directory(self, directory: Path) -> List[Path]:
        """Find all supported image files in a directory"""
        if not directory.exists() or not directory.is_dir():
            return []
        
        images = []
        for file_path in directory.iterdir():
            if file_path.is_file() and self.is_supported_image(file_path):
                images.append(file_path)
        
        return sorted(images)