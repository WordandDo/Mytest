import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional


def ensure_directory(path: Path) -> Path:
    """Ensure directory exists, create if it doesn't"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_url(string: str) -> bool:
    """Check if string is a valid URL"""
    return string.startswith(('http://', 'https://'))


def is_image_file(file_path: Path) -> bool:
    """Check if file is a supported image format"""
    supported_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tiff'}
    return file_path.suffix.lower() in supported_extensions


def save_json(data: Any, file_path: Path, indent: int = 2) -> None:
    """Save data to JSON file"""
    ensure_directory(file_path.parent)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_json(file_path: Path) -> Any:
    """Load data from JSON file"""
    if not file_path.exists():
        return None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to maximum length"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def print_search_results(results: List[Dict[str, str]], result_type: str = "search") -> None:
    """Pretty print search results"""
    if not results:
        print(f"No {result_type} results found.")
        return
    
    print(f"\n=== {result_type.title()} Results ({len(results)} found) ===")
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.get('title', 'No Title')}")
        
        if 'url' in result:
            print(f"   URL: {result['url']}")
        
        if 'link' in result:
            print(f"   Link: {result['link']}")
        
        if 'thumbnail' in result and result['thumbnail']:
            print(f"   Thumbnail: {result['thumbnail']}")
        
        if 'summary' in result:
            print(f"   Summary: {result['summary']}")
        
        # Print sources if available (for integrated summaries)
        if 'sources' in result and result['sources']:
            print(f"   Sources ({len(result['sources'])}):")
            for j, source in enumerate(result['sources'], 1):
                print(f"     {j}. {source.get('title', 'No Title')}")
                print(f"        {source.get('url', 'No URL')}")
        
        if 'error' in result:
            print(f"   Error: {result['error']}")


def validate_file_path(file_path: str) -> Path:
    """Validate and convert string to Path object"""
    path = Path(file_path).expanduser().resolve()
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    
    return path


def find_files_by_extension(directory: Path, extensions: List[str]) -> List[Path]:
    """Find all files with specified extensions in directory"""
    if not directory.exists() or not directory.is_dir():
        return []
    
    extensions_lower = [ext.lower() for ext in extensions]
    files = []
    
    for file_path in directory.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in extensions_lower:
            files.append(file_path)
    
    return sorted(files)


class ProgressBar:
    """Simple progress bar for console output"""
    
    def __init__(self, total: int, width: int = 50, desc: str = "Progress"):
        self.total = total
        self.width = width
        self.desc = desc
        self.current = 0
    
    def update(self, increment: int = 1, desc: Optional[str] = None):
        """Update progress bar"""
        self.current += increment
        if desc:
            self.desc = desc
        
        percent = self.current / self.total
        filled = int(self.width * percent)
        bar = '█' * filled + '-' * (self.width - filled)
        
        print(f'\r{self.desc}: |{bar}| {percent:.1%} ({self.current}/{self.total})', end='', flush=True)
        
        if self.current >= self.total:
            print()  # New line when complete


def setup_logging(log_file: Optional[Path] = None, level: str = "INFO") -> None:
    """Setup basic logging configuration"""
    import logging
    
    # Create logs directory if log_file is provided
    if log_file:
        ensure_directory(log_file.parent)
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )