import asyncio
import time
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Callable
from pan123 import Pan123

# 注意：这里需要确保 pan123 库已安装

from .config.settings import Config


class CloudStorageService:
    """Service for handling file uploads to 123Pan cloud storage"""
    
    def __init__(self):
        self.config = Config()
        self._pan_client: Optional[Pan123] = None
    
    @property
    def pan_client(self) -> Pan123:
        """Lazy initialization of Pan123 client"""
        if self._pan_client is None:
            access_token = self._get_access_token()
            self._pan_client = Pan123(access_token)
        return self._pan_client
    
    def _get_access_token(self) -> str:
        """Get access token directly from configuration"""
        token = self.config.PAN123_ACCESS_TOKEN
        if not token:
            raise ValueError(
                "PAN123_ACCESS_TOKEN is not set. Please set this environment variable "
                "with your 123Pan access token string."
            )
        return token
    
    def _extract_file_id(self, result: dict) -> str:
        """Extract file ID from upload result"""
        if not isinstance(result, dict):
            raise ValueError(f"Unexpected result type: {type(result)}")
        
        # Try different possible structures
        if "detail" in result and isinstance(result["detail"], dict) and "fileID" in result["detail"]:
            return result["detail"]["fileID"]
        
        if "fileID" in result:
            return result["fileID"]
        
        raise KeyError(f"fileID not found in result: {result}")
    
    def _ensure_trailing_slash_removed(self, url: str) -> str:
        """Remove trailing slash from URL"""
        return url.rstrip("/")
    
    async def upload_single_image(self, file_path: Path) -> Dict[str, str]:
        """
        Upload a single image file to cloud storage
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Dict containing name, fileID, and url
            
        Raises:
            RuntimeError: If upload fails after all retries
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() not in self.config.SUPPORTED_IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        last_error = None
        
        for attempt in range(1, self.config.MAX_RETRIES + 1):
            try:
                # Run the blocking upload in a thread pool
                result = await asyncio.to_thread(
                    self.pan_client.file.upload,
                    int(self.config.PAN123_PARENT_FILE_ID),
                    str(file_path)
                )
                
                file_id = self._extract_file_id(result)
                url = f"{self._ensure_trailing_slash_removed(self.config.PAN123_BASE_URL)}/{file_id}"
                
                return {
                    "name": file_path.name,
                    "fileID": file_id,
                    "url": url
                }
                
            except Exception as e:
                last_error = e
                print(f"[WARN] Upload failed ({file_path.name}), attempt {attempt}/{self.config.MAX_RETRIES}: {e}")
                
                if attempt < self.config.MAX_RETRIES:
                    await asyncio.sleep(self.config.RETRY_SLEEP)
        
        # All retries failed
        raise RuntimeError(f"Upload permanently failed for {file_path} — last error: {last_error}")
    
    async def upload_multiple_images(self, file_paths: List[Path], 
                                   progress_callback: Optional[Callable] = None) -> List[Dict[str, str]]:
        """
        Upload multiple image files to cloud storage
        
        Args:
            file_paths: List of file paths to upload
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of upload results
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
                # Continue with other files even if one fails
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