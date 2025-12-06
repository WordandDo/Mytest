# Unit tests for the search tool
import unittest
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

class TestImageSearch(unittest.TestCase):
    """Test cases for image search functionality"""
    
    def setUp(self):
        from services.image_search import ImageSearchService
        self.service = ImageSearchService()
    
    def test_url_detection(self):
        """Test URL detection functionality"""
        self.assertTrue(self.service._prepare_image_url("https://example.com/image.jpg"))
        self.assertTrue(self.service._prepare_image_url("http://example.com/image.png"))
    
    def test_result_deduplication(self):
        """Test result deduplication"""
        results = [
            {"link": "https://example1.com", "title": "Test 1", "thumbnail": "thumb1.jpg"},
            {"link": "https://example2.com", "title": "Test 2", "thumbnail": "thumb2.jpg"},
            {"link": "https://example1.com", "title": "Test 1 Duplicate", "thumbnail": "thumb1.jpg"},
        ]
        deduplicated = self.service._deduplicate_results(results)
        self.assertEqual(len(deduplicated), 2)
        self.assertEqual(deduplicated[0]["link"], "https://example1.com")
        self.assertEqual(deduplicated[1]["link"], "https://example2.com")


class TestTextSearch(unittest.TestCase):
    """Test cases for text search functionality"""
    
    def setUp(self):
        from services.text_search import TextSearchService
        self.service = TextSearchService()
    
    def test_api_key_validation(self):
        """Test API key validation"""
        # This will test the validation logic without making actual API calls
        try:
            self.service._validate_api_keys()
        except ValueError as e:
            # Expected if API keys are not set in test environment
            self.assertIn("Missing required API keys", str(e))


class TestCloudStorage(unittest.TestCase):
    """Test cases for cloud storage functionality"""
    
    def setUp(self):
        from services.cloud_storage import CloudStorageService
        self.service = CloudStorageService()
    
    def test_supported_image_formats(self):
        """Test supported image format detection"""
        from pathlib import Path
        
        test_files = [
            Path("test.jpg"),
            Path("test.png"),
            Path("test.gif"),
            Path("test.txt"),  # Not an image
            Path("test.pdf"),  # Not an image
        ]
        
        expected_results = [True, True, True, False, False]
        
        for file_path, expected in zip(test_files, expected_results):
            self.assertEqual(self.service.is_supported_image(file_path), expected)


class TestUtils(unittest.TestCase):
    """Test cases for utility functions"""
    
    def test_url_detection(self):
        """Test URL detection helper"""
        from utils.helpers import is_url
        
        self.assertTrue(is_url("https://example.com"))
        self.assertTrue(is_url("http://example.com"))
        self.assertFalse(is_url("not-a-url"))
        self.assertFalse(is_url("/path/to/file"))
    
    def test_text_truncation(self):
        """Test text truncation helper"""
        from utils.helpers import truncate_text
        
        text = "This is a long text that needs to be truncated"
        truncated = truncate_text(text, 20, "...")
        self.assertEqual(len(truncated), 20)
        self.assertTrue(truncated.endswith("..."))
    
    def test_file_size_formatting(self):
        """Test file size formatting"""
        from utils.helpers import format_file_size
        
        self.assertEqual(format_file_size(0), "0 B")
        self.assertEqual(format_file_size(1024), "1.0 KB")
        self.assertEqual(format_file_size(1024 * 1024), "1.0 MB")


if __name__ == '__main__':
    unittest.main()