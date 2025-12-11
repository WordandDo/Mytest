import asyncio
import aiohttp
import json
from typing import List, Dict, Optional
from urllib.parse import quote
from openai import OpenAI

from .config.settings import Config


class TextSearchService:
    """Service for handling text search (raw sources without LLM summarization)"""
    
    def __init__(self):
        self.config = Config()
        self._openai_client: Optional[OpenAI] = None
    
    @property
    def openai_client(self) -> OpenAI:
        """Lazy initialization of OpenAI client"""
        if self._openai_client is None:
            if not self.config.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is required for text search")
            
            self._openai_client = OpenAI(
                api_key=self.config.OPENAI_API_KEY,
                base_url=self.config.OPENAI_BASE_URL
            )
        return self._openai_client
    
    async def search_with_summaries(self, 
                                  query: str,
                                  k: int = None,
                                  region: Optional[str] = None,
                                  lang: Optional[str] = None,
                                  llm_model: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Perform text search and return raw sources (title + url) without Jina fetch or LLM summarization.
        """
        if k is None:
            k = self.config.DEFAULT_SEARCH_RESULTS
        if llm_model is None:
            llm_model = self.config.DEFAULT_LLM_MODEL
        
        # Only require SERPAPI for raw search
        if not self.config.SERPAPI_API_KEY:
            raise ValueError("SERPAPI_API_KEY is required for text search")
        
        async with aiohttp.ClientSession() as session:
            # Step 1: Get search results from SerpAPI
            search_results = await self._get_search_results(session, query, k, region, lang)
            
            # Directly return search results (title + url) without fetching page contents or LLM summarization
            return search_results
    
    def _validate_api_keys(self):
        """Validate that required API keys are available"""
        required_keys = {
            "SERPAPI_API_KEY": self.config.SERPAPI_API_KEY,
            "JINA_API_KEY": self.config.JINA_API_KEY,
            "OPENAI_API_KEY": self.config.OPENAI_API_KEY,
        }
        
        missing = [name for name, value in required_keys.items() if not value]
        if missing:
            raise ValueError(f"Missing required API keys: {', '.join(missing)}")
    
    async def _get_search_results(self, 
                                session: aiohttp.ClientSession,
                                query: str,
                                k: int,
                                region: Optional[str] = None,
                                lang: Optional[str] = None) -> List[Dict[str, str]]:
        """Get search results from SerpAPI"""
        params = {
            "engine": "google",
            "api_key": self.config.SERPAPI_API_KEY,
            "q": query,
            "num": min(k * 2, 10)  # Get more results in case some fail to fetch
        }
        
        if region:
            params["gl"] = region
        if lang:
            params["hl"] = lang
        
        try:
            async with session.get(
                self.config.SERPAPI_URL, 
                params=params, 
                timeout=self.config.REQUEST_TIMEOUT
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    print(f"[ERROR] SerpAPI HTTP {resp.status}: {error_text}")
                    return []
                
                data = await resp.json()
                
        except Exception as e:
            print(f"[ERROR] SerpAPI request failed: {e}")
            return []
        
        # Extract organic results
        organic_results = data.get("organic_results", [])
        candidates = []
        
        for item in organic_results:
            url = item.get("link")
            title = item.get("title", "")
            
            if url:
                candidates.append({"title": title, "url": url})
            
            if len(candidates) >= k:
                break
        
        print(f"[INFO] Found {len(candidates)} search results")
        return candidates
    
    async def _fetch_page_contents(self, 
                                 session: aiohttp.ClientSession,
                                 search_results: List[Dict[str, str]]) -> List[str]:
        """Fetch page content using Jina Reader"""
        async def fetch_single_page(result: Dict[str, str]) -> str:
            safe_url = quote(result["url"], safe=":/?&=#%+-~._")
            jina_url = f"{self.config.JINA_BASE}{safe_url}"
            headers = {"Authorization": f"Bearer {self.config.JINA_API_KEY}"}
            
            try:
                async with session.get(
                    jina_url, 
                    headers=headers, 
                    timeout=self.config.REQUEST_TIMEOUT
                ) as resp:
                    if resp.status == 200:
                        content = await resp.text()
                        print(f"[INFO] Successfully fetched content from {result['url']}")
                        return content
                    else:
                        error_text = await resp.text()
                        print(f"[WARN] Jina Reader HTTP {resp.status} for {result['url']}: {error_text}")
                        
            except Exception as e:
                print(f"[WARN] Failed to fetch content from {result['url']}: {e}")
            
            return ""
        
        # Fetch all pages concurrently
        return await asyncio.gather(*[fetch_single_page(result) for result in search_results])
    
    async def _generate_integrated_summary(self,
                                         query: str,
                                         search_results: List[Dict[str, str]],
                                         page_contents: List[str],
                                         model: str) -> str:
        """Generate integrated AI summary from all search results"""
        # Combine all page contents with source information
        combined_content = ""
        for i, (result, content) in enumerate(zip(search_results, page_contents), 1):
            if content.strip():  # Only include non-empty content
                combined_content += f"\n=== 来源 {i}: {result['title']} ({result['url']}) ===\n"
                combined_content += content[:self.config.MAX_SUMMARY_CHARS // len(search_results)]
                combined_content += "\n"
        
        if not combined_content.strip():
            return "无法获取有效内容进行摘要。"
        
        return await self._llm_summarize_integrated_async(
            query=query,
            combined_content=combined_content,
            model=model
        )
    
    def _create_summarized_passages(self, 
                                  search_results: List[Dict[str, str]], 
                                  integrated_summary: str) -> List[Dict[str, str]]:
        """Create summarized passages linked to their respective sources"""
        return [{
            "title": "Integrated Search Summary",
            "url": f"Based on {len(search_results)} sources",
            "summary": integrated_summary,
            "sources": [{"title": r["title"], "url": r["url"]} for r in search_results]
        }]
    
    async def _llm_summarize_integrated_async(self,
                                            query: str,
                                            combined_content: str,
                                            model: str = None,
                                            temperature: float = None) -> str:
        """Generate integrated AI summary from combined content using OpenAI-compatible API"""
        if model is None:
            model = self.config.DEFAULT_LLM_MODEL
        if temperature is None:
            temperature = self.config.DEFAULT_TEMPERATURE
        
        system_prompt = (
            "你是一个严谨的学术研究助手。请根据用户查询，综合分析所有提供的网页内容，生成一个全面且结构化的中文摘要报告：\n"
            "要求：\n"
            "1) 紧扣用户查询的核心意图，提取最相关的信息\n"
            "2) 综合所有来源的观点，形成完整的知识图景\n"
            "3) 保持客观中立的学术态度\n"
            "4) 结构清晰，分点论述，每个要点简洁明了\n"
            "5) 如有具体数据、时间、研究结论等关键信息，请明确标注\n"
            "6) 如发现不同来源间存在分歧或互补信息，请指出\n"
            "7) 控制篇幅在10-15句话内，突出重点"
        )
        
        user_content = (
            f"【用户查询】{query}\n\n"
            f"【综合内容来源】\n{combined_content}"
        )
        
        def _sync_call():
            try:
                response = self.openai_client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                )
                
                # === DEBUG LOG START ===
                # 打印响应类型和内容预览，帮助排查 API 返回 HTML 的问题
                print(f"\n[DEBUG] LLM Response Type: {type(response)}")
                if isinstance(response, str):
                    print(f"[DEBUG] Raw String Response (First 1000 chars):\n{response[:1000]}")
                    print("-" * 60)
                # === DEBUG LOG END ===

                # 兼容性处理
                if isinstance(response, str):
                    if response.strip().startswith("{"):
                        try:
                            data = json.loads(response)
                            if isinstance(data, dict) and "choices" in data:
                                return data["choices"][0]["message"]["content"].strip()
                        except:
                            pass
                    return response.strip()
                
                elif isinstance(response, dict):
                    if "choices" in response:
                         msg = response["choices"][0].get("message", {})
                         if isinstance(msg, dict):
                             return msg.get("content", "").strip()
                         return msg.content.strip()

                return response.choices[0].message.content.strip()
                
            except Exception as e:
                print(f"[ERROR] Integrated LLM summarization failed: {e}")
                import traceback
                traceback.print_exc()
                return f"（综合摘要生成失败：{e}）"
        
        # Run the synchronous OpenAI call in a thread pool
        return await asyncio.to_thread(_sync_call)
    
    async def _llm_summarize_async(self,
                                 query: str,
                                 page_title: str,
                                 page_url: str,
                                 page_text: str,
                                 model: str = None,
                                 temperature: float = None) -> str:
        """Generate AI summary using OpenAI-compatible API"""
        if model is None:
            model = self.config.DEFAULT_LLM_MODEL
        if temperature is None:
            temperature = self.config.DEFAULT_TEMPERATURE
        
        system_prompt = (
            "你是一个严谨的学术摘要助手。请根据\"用户查询\"对\"网页正文\"进行高度相关的中文摘要：\n"
            "要求：1) 紧扣查询意图；2) 客观中立；3) 结构清晰≤6句；"
            "4) 如有数据/时间/结论请明确给出；5) 若正文相关性弱，请简要说明。"
        )
        
        user_content = (
            f"【用户查询】{query}\n"
            f"【网页标题】{page_title}\n"
            f"【网页链接】{page_url}\n"
            f"【网页正文（截断）】\n{page_text[:self.config.MAX_SUMMARY_CHARS]}"
        )
        
        def _sync_call():
            try:
                response = self.openai_client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                )

                # === DEBUG LOG START ===
                # 打印响应类型
                # print(f"[DEBUG] Single LLM Response Type: {type(response)}")
                if isinstance(response, str):
                    print(f"[DEBUG] Raw String Response (First 500 chars):\n{response[:500]}")
                # === DEBUG LOG END ===

                if isinstance(response, str):
                    if response.strip().startswith("{"):
                        try:
                            data = json.loads(response)
                            if isinstance(data, dict) and "choices" in data:
                                return data["choices"][0]["message"]["content"].strip()
                        except:
                            pass
                    return response.strip()
                
                elif isinstance(response, dict):
                    if "choices" in response:
                         msg = response["choices"][0].get("message", {})
                         if isinstance(msg, dict):
                             return msg.get("content", "").strip()
                         return msg.content.strip()
                
                return response.choices[0].message.content.strip()

            except Exception as e:
                print(f"[ERROR] LLM summarization failed: {e}")
                return f"（摘要生成失败：{e}）"
        
        # Run the synchronous OpenAI call in a thread pool
        return await asyncio.to_thread(_sync_call)
    
    async def batch_search(self, 
                          queries: List[str],
                          k: int = None,
                          region: Optional[str] = None,
                          lang: Optional[str] = None,
                          llm_model: Optional[str] = None) -> List[List[Dict[str, str]]]:
        """
        Perform multiple text searches concurrently
        """
        tasks = [
            self.search_with_summaries(query, k, region, lang, llm_model)
            for query in queries
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)
