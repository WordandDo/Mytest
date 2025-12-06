import json
import os
from datetime import datetime
from pathlib import Path
from serpapi import GoogleSearch

# ===== 配置选项 =====
# 选择1: 自动识别（使用图片反向搜索，AI 自动生成查询词）
USE_AUTO_QUERY = True  # ✅ 反向图片搜索（设为 True）
# False = 手动查询词搜索

if USE_AUTO_QUERY:
    # 自动识别模式 - 上传图片，AI 自动识别并生成查询词
    params = {
        "engine": "google_reverse_image",
        "image_url": "https://youke1.picui.cn/s1/2025/10/27/68ff3e07492f5.png",
        "api_key": "3df0d1d00b37f25a5e7ec12d40cde4845284f8986ea09aab6a77b49577507c2a",
        "gl": "us",
        "hl": "en"
    }
else:
    # 手动查询模式 - 自己指定查询词
    params = {
        "engine": "google_images",  # 或 "google" 或 "google_shopping"
        "q": "cake",    # ← 手动指定查询词
        "tbm": "isch",              # 图片搜索（仅 google_images 需要）
        "api_key": "3df0d1d00b37f25a5e7ec12d40cde4845284f8986ea09aab6a77b49577507c2a",
        "gl": "us",
        "hl": "en"
    }

search = GoogleSearch(params)
results = search.get_dict()

# 提取搜索信息
search_info = results.get("search_information", {})
query = search_info.get("query_displayed", params.get('q', 'Unknown'))
total_results = search_info.get("total_results", 0)

print(f"搜索模式: {'🔍 AI自动识别' if USE_AUTO_QUERY else '✍️ 手动查询'}")
if USE_AUTO_QUERY:
    print(f"原始图片: {params.get('image_url', 'N/A')}")
    print(f"AI识别查询: {query}")
else:
    print(f"手动查询词: {params.get('q', 'N/A')}")
    print(f"显示查询: {query}")
print(f"总结果数: {total_results}")
print("-" * 80)

# 打印完整的搜索参数信息
print("\n[搜索参数详情]")
print(f"Engine: {params['engine']}")
print(f"Country (gl): {params.get('gl', 'default')}")
print(f"Language (hl): {params.get('hl', 'default')}")
print("-" * 80)

# 提取搜索结果（根据不同引擎）
if params['engine'] == 'google_reverse_image':
    image_results = results.get("image_results", [])
    result_type = "图片搜索结果"
elif params['engine'] == 'google_images':
    image_results = results.get("images_results", [])
    result_type = "图片"
elif params['engine'] == 'google':
    image_results = results.get("organic_results", [])
    result_type = "网页"
elif params['engine'] == 'google_shopping':
    image_results = results.get("shopping_results", [])
    result_type = "商品"
else:
    image_results = []

print(f"找到 {len(image_results)} 个{result_type}:\n")

# 组织要保存的数据
output_data = {
    "timestamp": datetime.now().isoformat(),
    "search_query": query,
    "total_results": total_results,
    "search_mode": "reverse_image" if USE_AUTO_QUERY else "manual_query",
    "image_url": params.get("image_url") if USE_AUTO_QUERY else None,
    "manual_query": params.get("q") if not USE_AUTO_QUERY else None,
    "num_results_returned": len(image_results),
    "results": []
}

for idx, result in enumerate(image_results, 1):
    print(f"结果 {idx}:")
    print(f"  标题: {result.get('title', 'N/A')}")
    print(f"  链接: {result.get('link', 'N/A')}")
    print(f"  摘要: {result.get('snippet', 'N/A')}")
    if 'thumbnail' in result:
        print(f"  缩略图: {result.get('thumbnail')}")
    print()
    
    # 保存结构化的结果
    output_data["results"].append({
        "position": result.get("position", idx),
        "title": result.get("title"),
        "link": result.get("link"),
        "snippet": result.get("snippet"),
        "thumbnail": result.get("thumbnail"),
        "displayed_link": result.get("displayed_link"),
        "source": result.get("source"),
        "redirect_link": result.get("redirect_link"),
        "favicon": result.get("favicon")
    })

# 创建输出目录
output_dir = Path("compare/results")
output_dir.mkdir(parents=True, exist_ok=True)

# 保存为 JSON 文件
output_filename = f"serpapi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
output_path = output_dir / output_filename

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"\n{'='*80}")
print(f"✅ 结果已保存到: {output_path}")
print(f"   查询: {query}")
print(f"   找到: {len(image_results)} 个结果")
print(f"{'='*80}")