import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    # API Keys
    SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "")
    JINA_API_KEY = os.getenv("JINA_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.chatanywhere.tech/v1")

    # SerpAPI Configuration
    SERPAPI_URL = "https://serpapi.com/search"

    # Jina Reader Configuration
    JINA_BASE = "https://r.jina.ai/"

    # File and Path Configuration
    # 指向 src/utils/search_v2
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    UPLOADS_DIR = PROJECT_ROOT / "uploads"
    LOGS_DIR = PROJECT_ROOT / "logs"

    # 123Pan Configuration
    PAN123_BASE_URL = os.getenv("PAN123_BASE_URL", "https://vip.123pan.cn/1820855797")
    PAN123_PARENT_FILE_ID = os.getenv(
        "PAN123_PARENT_FILE_ID", "yk6baz03t0l000d7w33fl91aiec2g63tDIYvDdYyAIU2AvxPDqYw"
    )
    # 使用基于 PROJECT_ROOT 的绝对路径
    PAN123_ACCESS_TOKEN_FILE = os.getenv(
        "PAN123_ACCESS_TOKEN_FILE", str(PROJECT_ROOT / "config" / "access_token.txt")
    )

    # Image Configuration
    SUPPORTED_IMAGE_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".bmp",
        ".tiff",
    }

    # Search Configuration
    DEFAULT_SEARCH_RESULTS = 5
    DEFAULT_IMAGE_RESULTS = 6

    # LLM Configuration
    DEFAULT_LLM_MODEL = "gpt-4o-mini"
    DEFAULT_TEMPERATURE = 0.2
    MAX_SUMMARY_CHARS = 200

    # Retry Configuration
    MAX_RETRIES = 3
    RETRY_SLEEP = 1.5

    # Timeout Configuration
    REQUEST_TIMEOUT = 60

    @classmethod
    def validate_required_keys(cls):
        """Validate that all required API keys are present"""
        required_keys = {
            "SERPAPI_API_KEY": cls.SERPAPI_API_KEY,
            "JINA_API_KEY": cls.JINA_API_KEY,
            "OPENAI_API_KEY": cls.OPENAI_API_KEY,
        }

        missing = [name for name, value in required_keys.items() if not value]
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}"
            )

        return True

    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        cls.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)