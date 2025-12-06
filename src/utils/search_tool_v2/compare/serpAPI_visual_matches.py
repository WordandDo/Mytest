"""
纯视觉相似度搜索 - 使用 google_lens 引擎
返回 visual_matches 字段：只基于图像特征匹配，不依赖文本查询
"""
import json
from datetime import datetime
from pathlib import Path
from serpapi import GoogleSearch

# ===== 使用 google_lens 引擎 - 纯视觉匹配 =====
params = {
    "engine": "google_lens",  # ✅ 使用 Lens 引擎
    "url": "https://youke1.picui.cn/s1/2025/10/27/68ff3e07492f5.png",  # 图片URL（参数名是 url）
    "api_key": "3df0d1d00b37f25a5e7ec12d40cde4845284f8986ea09aab6a77b49577507c2a",
    "gl": "us",
    "hl": "en"
}

print("="*80)
print("🔍 纯视觉相似度搜索 (Google Lens)")
print("="*80)
print(f"引擎: google_lens")
print(f"图片: {params['url']}")
print(f"特点: 基于图像特征匹配，不依赖文本查询")
print("-"*80)

search = GoogleSearch(params)
results = search.get_dict()

# 提取视觉匹配结果（纯图像相似度）
visual_matches = results.get("visual_matches", [])
regular_results = results.get("results", [])

print(f"\n✅ 视觉匹配结果: {len(visual_matches)} 个")
print(f"📄 常规结果: {len(regular_results)} 个\n")

# 显示视觉匹配结果
if visual_matches:
    print("="*80)
    print("🎨 视觉匹配图片（按相似度排序）")
    print("="*80)
    
    for idx, match in enumerate(visual_matches[:10], 1):  # 显示前10个
        print(f"\n匹配 {idx}:")
        print(f"  来源: {match.get('source', 'N/A')}")
        print(f"  标题: {match.get('title', 'N/A')}")
        print(f"  链接: {match.get('link', 'N/A')}")
        if 'thumbnail' in match:
            print(f"  图片: {match.get('thumbnail')}")
else:
    print("⚠️  未找到 visual_matches 结果")
    
    # 如果没有 visual_matches，尝试其他字段
    print("\n尝试其他结果字段...")
    
    # 检查是否有其他结果
    all_keys = results.keys()
    print(f"可用字段: {list(all_keys)[:10]}")
    
    # 尝试提取任何包含图片的结果
    for key in ['results', 'serpapi_pagination']:
        if key in results:
            print(f"\n{key}: {len(results[key]) if isinstance(results[key], list) else 'exists'}")

# 保存结果
output_dir = Path("compare/results")
output_dir.mkdir(parents=True, exist_ok=True)

output_data = {
    "timestamp": datetime.now().isoformat(),
    "engine": "google_lens",
    "search_type": "visual_matches",
    "image_url": params['url'],
    "visual_matches_count": len(visual_matches),
    "results_count": len(regular_results),
    "visual_matches": visual_matches,
    "all_keys": list(results.keys())  # 调试用
}

output_filename = f"lens_visual_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
output_path = output_dir / output_filename

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"\n{'='*80}")
print(f"✅ 结果已保存到: {output_path}")
print(f"{'='*80}")

