import asyncio
import argparse
import sys
from pathlib import Path
from typing import List, Union

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import Config
from services.image_search import ImageSearchService
from services.text_search import TextSearchService
from utils.helpers import print_search_results, validate_file_path, is_url, ProgressBar


class SearchToolApp:
    """Main application class for the search tool"""

    def __init__(self):
        self.config = Config()
        self.image_search = ImageSearchService()
        self.text_search = TextSearchService()

    async def search_images(
        self, image_inputs: List[str], k: int = None, save_results: bool = False
    ) -> List[List[dict]]:
        """
        Search for similar images

        Args:
            image_inputs: List of image URLs or file paths
            k: Number of results per image
            save_results: Whether to save results to file

        Returns:
            List of search results for each input
        """
        if k is None:
            k = self.config.DEFAULT_IMAGE_RESULTS

        print(f"Starting image search for {len(image_inputs)} image(s)...")

        # Process inputs - validate file paths
        processed_inputs = []
        for img_input in image_inputs:
            if is_url(img_input):
                processed_inputs.append(img_input)
            else:
                try:
                    file_path = validate_file_path(img_input)
                    processed_inputs.append(file_path)
                except (FileNotFoundError, ValueError) as e:
                    print(f"[ERROR] {e}")
                    continue

        if not processed_inputs:
            print("[ERROR] No valid image inputs found")
            return []

        # Perform searches
        all_results = []
        for i, img_input in enumerate(processed_inputs, 1):
            print(
                f"\n[INFO] ({i}/{len(processed_inputs)}) Searching image: {img_input}"
            )

            try:
                results = await self.image_search.search_by_image(img_input, k=k)
                all_results.append(results)

                print_search_results(results, "Image Search")

            except Exception as e:
                print(f"[ERROR] Image search failed: {e}")
                all_results.append([])

        # Save results if requested
        if save_results:
            self._save_search_results(all_results, "image_search_results.json")

        return all_results

    async def search_text(
        self,
        queries: List[str],
        k: int = None,
        region: str = None,
        lang: str = None,
        llm_model: str = None,
        save_results: bool = False,
    ) -> List[List[dict]]:
        """
        Search text queries and generate summaries

        Args:
            queries: List of search queries
            k: Number of results per query
            region: Search region
            lang: Search language
            llm_model: LLM model for summaries
            save_results: Whether to save results to file

        Returns:
            List of search results for each query
        """
        if k is None:
            k = self.config.DEFAULT_SEARCH_RESULTS
        if llm_model is None:
            llm_model = self.config.DEFAULT_LLM_MODEL

        print(f"Starting text search for {len(queries)} query(ies)...")

        # Perform searches
        all_results = []
        for i, query in enumerate(queries, 1):
            print(f'\n[INFO] ({i}/{len(queries)}) Searching: "{query}"')

            try:
                results = await self.text_search.search_with_summaries(
                    query=query, k=k, region=region, lang=lang, llm_model=llm_model
                )
                all_results.append(results)

                print_search_results(results, "Text Search")

            except Exception as e:
                print(f"[ERROR] Text search failed: {e}")
                all_results.append([])

        # Save results if requested
        if save_results:
            self._save_search_results(all_results, "text_search_results.json")

        return all_results

    def _save_search_results(self, results: List[List[dict]], filename: str):
        """Save search results to JSON file"""
        from utils.helpers import save_json

        output_path = self.config.LOGS_DIR / filename
        save_json(results, output_path)
        print(f"\n[INFO] Results saved to: {output_path}")

    async def run_demo(self):
        """Run demo searches"""
        print("=== Search Tool Demo ===\n")

        # Demo image search
        print("1. Image Search Demo")
        demo_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8f/Andrew_Yao_P1130016_%28cropped%29.jpg/500px-Andrew_Yao_P1130016_%28cropped%29.jpg"
        await self.search_images([demo_image_url], k=3)

        print("\n" + "=" * 50 + "\n")

        # Demo text search
        print("2. Text Search Demo")
        demo_query = "OpenAI GPT-4 capabilities and features"
        await self.search_text([demo_query], k=3, region="us", lang="en")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Advanced Search Tool with Image and Text Search Capabilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Image search with URLs
  python main.py image-search "https://example.com/image.jpg" --results 5
  
  # Image search with local files
  python main.py image-search "/path/to/image.jpg" "/path/to/image2.png"
  
  # Text search
  python main.py text-search "OpenAI GPT-4 features" --results 5 --region us --lang en
  
  # Multiple text searches
  python main.py text-search "AI developments 2024" "Machine learning trends" --llm-model gpt-4
  
  # Run demo
  python main.py demo
        """,
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Image search command
    img_parser = subparsers.add_parser("image-search", help="Search for similar images")
    img_parser.add_argument("images", nargs="+", help="Image URLs or file paths")
    img_parser.add_argument(
        "--results", "-k", type=int, default=None, help="Number of results per image"
    )
    img_parser.add_argument(
        "--save", action="store_true", help="Save results to JSON file"
    )

    # Text search command
    text_parser = subparsers.add_parser(
        "text-search", help="Search text with AI summaries"
    )
    text_parser.add_argument("queries", nargs="+", help="Search queries")
    text_parser.add_argument(
        "--results", "-k", type=int, default=None, help="Number of results per query"
    )
    text_parser.add_argument(
        "--region", "-r", type=str, default=None, help="Search region (e.g., us, cn)"
    )
    text_parser.add_argument(
        "--lang", "-l", type=str, default=None, help="Search language (e.g., en, zh-CN)"
    )
    text_parser.add_argument(
        "--llm-model", "-m", type=str, default=None, help="LLM model for summaries"
    )
    text_parser.add_argument(
        "--save", action="store_true", help="Save results to JSON file"
    )

    # Demo command
    subparsers.add_parser("demo", help="Run demonstration searches")

    return parser


async def main():
    """Main application entry point"""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Initialize configuration
    try:
        Config.validate_required_keys()
        Config.create_directories()
    except ValueError as e:
        print(f"[ERROR] Configuration error: {e}")
        print("Please check your .env file and ensure all required API keys are set.")
        sys.exit(1)

    # Initialize app
    app = SearchToolApp()

    # Handle commands
    try:
        if args.command == "image-search":
            await app.search_images(
                image_inputs=args.images, k=args.results, save_results=args.save
            )

        elif args.command == "text-search":
            await app.search_text(
                queries=args.queries,
                k=args.results,
                region=args.region,
                lang=args.lang,
                llm_model=args.llm_model,
                save_results=args.save,
            )

        elif args.command == "demo":
            await app.run_demo()

        else:
            parser.print_help()

    except KeyboardInterrupt:
        print("\n[INFO] Operation cancelled by user")
    except Exception as e:
        print(f"[ERROR] Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
