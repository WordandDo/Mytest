from typing import Union, List, Dict, Any
import requests
import os
import pdb
import bdb

# os.environ['SERPER_API_KEY'] = ''

class WebSearchTool:
    name = "web_search"
    description = (
        "A web search tool powered by Serper API that can search the internet for current information. "
        "Use this tool to find recent news, articles, websites, and general information from the web."
    )
    parameters = [
        {
            'name': 'queries',
            'type': 'array',
            'array_type': 'string', 
            'description': 'Array of search queries to execute in parallel. Each query returns top-k results.',
            'required': True
        }
    ]

    def __init__(self, top_k=3, search_type="search", max_workers=5):
        """
        Initialize WebSearchTool with configurable parameters
        
        Args:
            top_k: Number of top search results to return per query (default: 3, max: 10)
            search_type: Type of search - "search", "news", or "images" (default: "search")
            max_workers: Maximum number of concurrent workers for parallel requests (default: 5)
        """
        self.api_key = os.getenv('SERPER_API_KEY')
        if not self.api_key:
            print("Warning: SERPER_API_KEY environment variable not set")
        self.top_k = min(top_k, 10)
        self.search_type = search_type
        self.max_workers = max_workers

    def call(self, params: Union[str, dict]) -> str:
        try:
            queries = params.get("queries")
            if not queries:
                return "[WebSearch] Error: Queries parameter is required"
            
            # Handle single query as string
            if isinstance(queries, str):
                queries = [queries]
            elif not isinstance(queries, list):
                return "[WebSearch] Error: Queries must be a string or array of strings"
            
            if not self.api_key:
                return "[WebSearch] Error: SERPER_API_KEY not configured"
            
            # Process all queries in parallel
            return self._search_queries(queries)
                
        except Exception as e:
            return f"[WebSearch] Error: {str(e)}"

    def _search_queries(self, queries: List[str]) -> str:
        """Search multiple queries in parallel"""
        import concurrent.futures
        
        def search_single_query(query):
            try:
                # Make API request to Serper
                url = f"https://google.serper.dev/{self.search_type}"
                headers = {
                    'X-API-KEY': self.api_key,
                    'Content-Type': 'application/json'
                }
                
                payload = {
                    'q': query,
                    'num': self.top_k
                }
                
                response = requests.post(url, headers=headers, json=payload, timeout=10)
                response.raise_for_status()

                
                data = response.json()
                results = self._format_results(data, query, self.search_type)
                
                return {
                    'query': query,
                    'success': True,
                    'results': results
                }
                
            except Exception as e:
                if isinstance(e, bdb.BdbQuit):
                    raise e
                return {
                    'query': query,
                    'success': False,
                    'error': str(e)
                }
        
        # Process queries in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            search_results = list(executor.map(search_single_query, queries))
        
        # Format combined results as string
        return self._format_combined_results(search_results)

    def _format_results(self, data: Dict[str, Any], query: str, search_type: str) -> List[Dict[str, str]]:
        """Format the search results as a list of dictionaries for programmatic use"""
        
        results = []
        
        if search_type == "search":
            # Handle organic search results
            organic = data.get('organic', [])
            for result in organic:
                results.append({
                    'query': query,
                    'url': result.get('link', ''),
                    'title': result.get('title', ''),
                    'snippet': result.get('snippet', '')
                })
        
        elif search_type == "news":
            # Handle news results
            news = data.get('news', [])
            for article in news:
                results.append({
                    'query': query,
                    'url': article.get('link', ''),
                    'title': article.get('title', ''),
                    'snippet': article.get('snippet', ''),
                    'date': article.get('date', ''),
                    'source': article.get('source', '')
                })
        
        elif search_type == "images":
            # Handle image results
            images = data.get('images', [])
            for image in images:
                results.append({
                    'query': query,
                    'url': image.get('imageUrl', ''),
                    'title': image.get('title', ''),
                    'snippet': f"Image from {image.get('source', '')}" if image.get('source', '') else ''
                })
        
        # Only need url and snippet
        results = [{'url': item.get('url', ''), 'snippet': item.get('snippet', '')} for item in results]
        return results

    def _format_combined_results(self, search_results: List[Dict]) -> str:
        """Format all search results into a single string with URLs and snippets"""
        if not search_results:
            return "[WebSearch] No queries processed"
        
        output = ""
        successful_queries = [r for r in search_results if r['success']]
        
        # Extract all results from successful queries
        all_results = []
        for query_result in successful_queries:
            if 'results' in query_result and query_result['results']:
                all_results.extend(query_result['results'])
        
        # Format results as simple numbered list
        for i, item in enumerate(all_results, 1):
            url = item.get('url', '')
            snippet = item.get('snippet', '')
            if url:  # Only show items with valid URLs
                output += f"{i}.\nurl: {url}\nsnippet: {snippet}\n\n"
        
        return output


if __name__ == '__main__':
    # 测试WebSearchTool工具
    search_tool = WebSearchTool(top_k=3, search_type="search")
    
    print("=== WebSearchTool Test Cases ===\n")
    
    # 测试用例1: 单个查询
    print("Test 1: Single query")
    result = search_tool.call({
        "queries": ["Python programming tutorial"]
    })
    print(result)
    print("-" * 50)
    
    # 测试用例2: 多个查询并行搜索
    print("Test 2: Multiple queries parallel search")
    result = search_tool.call({
        "queries": [
            "machine learning basics",
            "deep learning frameworks"
        ]
    })
    print(result)
    print("-" * 50)
    
    # # 测试用例3: 参数验证
    # print("Test 3: Parameter validation")
    # result = search_tool.call({})  # 缺少queries
    # print(result)
    
    # print(f"Tool configuration: top_k={search_tool.top_k}, search_type={search_tool.search_type}")
