import asyncio
import os
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

try:
    from src.utils.search_v2 import TextSearchService, ImageSearchService
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

async def test_text_search():
    print("\n" + "="*50)
    print("🔍 测试文本搜索与 AI 摘要 (TextSearchService)")
    print("="*50)
    
    # 检查必要的 API Key
    required_keys = ["SERPAPI_API_KEY", "JINA_API_KEY", "OPENAI_API_KEY"]
    missing = [k for k in required_keys if not os.getenv(k)]
    if missing:
        print(f"⚠️  跳过测试: 缺少环境变量 {', '.join(missing)}")
        return

    try:
        service = TextSearchService()
        query = "2024年诺贝尔物理学奖得主是谁"
        print(f"❓ 查询问题: {query}")
        print("⏳ 正在搜索、抓取内容并生成摘要 (这可能需要几秒钟)...")
        
        # 执行搜索
        results = await service.search_with_summaries(query, k=3)
        
        if results:
            summary_item = results[0]
            print("\n✅ 综合摘要:")
            print("-" * 30)
            print(summary_item.get('summary'))
            print("-" * 30)
            print("\n🔗 参考来源:")
            for idx, source in enumerate(summary_item.get('sources', []), 1):
                print(f"   {idx}. {source['title'][:30]}... ({source['url']})")
        else:
            print("⚠️  未找到结果。")
            
    except Exception as e:
        print(f"❌ 文本搜索测试失败: {e}")

async def test_image_search():
    print("\n" + "="*50)
    print("🖼️  测试以图搜图 (ImageSearchService)")
    print("="*50)
    
    if not os.getenv("SERPAPI_API_KEY"):
        print(f"⚠️  跳过测试: 缺少环境变量 SERPAPI_API_KEY")
        return

    try:
        service = ImageSearchService()
        # 使用 Python Logo 作为测试图片
        image_url = "https://www.python.org/static/community_logos/python-logo-master-v3-TM.png"
        print(f"🔎 正在反向搜索图片: {image_url}")
        
        results = await service.search_by_image(image_url, k=3)
        
        if results:
            print(f"\n✅ 找到 {len(results)} 个相关结果:")
            for idx, res in enumerate(results, 1):
                print(f"   {idx}. [{res.get('title')}]")
                print(f"      链接: {res.get('link')}")
        else:
            print("⚠️  未找到相似图片结果。")
            
    except Exception as e:
        print(f"❌ 图片搜索测试失败: {e}")

if __name__ == "__main__":
    # 加载环境变量
    from dotenv import load_dotenv
    load_dotenv()
    
    asyncio.run(test_text_search())
    asyncio.run(test_image_search())