import asyncio
import base64
import json
import io
import sys
import os
import time

# 确保可以将 src 目录加入路径以导入项目模块
sys.path.append(os.getcwd())

try:
    from src.utils.mcp_sse_client import MCPSSEClient
    from PIL import Image
except ImportError:
    print("❌ Error: 缺少必要的库。请确保安装了 Pillow 并在项目根目录运行。")
    print("pip install Pillow requests aiohttp")
    sys.exit(1)

# =============================================================================
# 辅助函数：生成测试用的 Base64 图片
# =============================================================================
def create_dummy_base64_image(color=(255, 0, 0), size=(200, 200), text="A"):
    """生成一张带颜色的测试图片"""
    img = Image.new('RGB', size, color)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# =============================================================================
# 核心演示流程
# =============================================================================
async def main():
    server_url = "http://localhost:8080"
    print(f"🔌 正在连接 MCP Server: {server_url}/sse ...")
    
    client = MCPSSEClient(f"{server_url}/sse")
    await client.connect()
    
    try:
        print("\n" + "="*60)
        print("🎨 场景 1: 准备数据与初始裁切 (Image Cropping)")
        print("="*60)
        
        # 1. 构造包含图片的对话历史
        # 模拟：用户上传了一张红色图片，并标记为 <img_red>
        b64_img = create_dummy_base64_image(color=(200, 50, 50), size=(400, 400))
        
        conversation_history = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "看这张红色的测试图 <img_target>"},
                    {"type": "image_url", "image_url": {"url": b64_img}}
                ]
            }
        ]

        # 2. 调用裁切工具
        # 注意：存储模式现在由服务端的 SEARCH_STORAGE_MODE 环境变量控制
        crop_args = {
            "crop_config": {"img_target": [0, 0, 100, 100]},
            "messages": conversation_history
        }
        
        print(">> 调用工具: crop_images_by_token")
        crop_result_raw = await client.call_tool("crop_images_by_token", crop_args)
        
        # 解析结果
        crop_result = parse_mcp_result(crop_result_raw)
        print(f"✅ 裁切成功! 结果: {json.dumps(crop_result, indent=2, ensure_ascii=False)}")
        
        # 获取裁切后的图片路径（假设工具返回的是本地路径）
        cropped_image_path = crop_result.get("img_target")
        if not cropped_image_path or "Error" in cropped_image_path:
            print("❌ 裁切失败，无法继续后续演示。")
            return

        print("\n" + "="*60)
        print("🔍 场景 2: 连通性测试 - 使用裁切结果进行搜图 (Reverse Search)")
        print("="*60)
        
        # 3. 将裁切结果传给反向搜图工具
        # 注意：Search Server 的 reverse_image_search 接受 image_url 参数
        # 如果是本地路径，最好转为 file:// 协议或者直接传路径（取决于您的 search_server 实现是否支持路径）
        # 这里直接传路径演示
        
        search_args = {
            "image_url": cropped_image_path,
            "k": 1
        }
        
        print(f">> 调用工具: reverse_image_search")
        print(f"   输入: {cropped_image_path}")
        
        # 这里可能会因为没有真实的搜图后端而报错/返回空，但能证明调用链路通了
        search_result_raw = await client.call_tool("reverse_image_search", search_args)
        print(f"✅ 调用完成 (模拟): {parse_mcp_result(search_result_raw)}")

        print("\n" + "="*60)
        print("🛡️ 场景 3: 安全机制验证 - 禁止递归裁切 (Anti-Recursive Check)")
        print("="*60)
        
        # 4. 尝试对“裁切出来的图片”再次进行裁切
        # 构造一个新的对话上下文，引用刚才生成的裁切图
        recursive_history = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "对刚才的裁切结果再切一次 <img_recursive>"},
                    {"type": "image_url", "image_url": {"url": cropped_image_path}} # 这里的路径包含 mcp_derived_crop_
                ]
            }
        ]
        
        recursive_args = {
            "crop_config": {"img_recursive": [0, 0, 50, 50]},
            "messages": recursive_history
        }
        
        print(f">> 尝试再次裁切受保护的图片: {os.path.basename(cropped_image_path)}")
        recursive_result_raw = await client.call_tool("crop_images_by_token", recursive_args)
        recursive_result = parse_mcp_result(recursive_result_raw)
        
        print(f"✅ 结果 (预期应报错):")
        print(json.dumps(recursive_result, indent=2, ensure_ascii=False))
        
        if "Error" in str(recursive_result) and "recursive" in str(recursive_result):
            print("\n🎉 验证通过：成功拦截了二次裁切请求！")
        else:
            print("\n⚠️ 验证警告：未检测到预期的拦截错误信息。")

        print("\n" + "="*60)
        print("🌐 场景 4: 文本搜索 (Web Search)")
        print("="*60)
        
        web_args = {"query": "MCP Model Context Protocol", "k": 1}
        print(f">> 调用工具: web_search ('{web_args['query']}')")
        web_res = await client.call_tool("web_search", web_args)
        print(f"✅ 搜索结果摘要: {str(parse_mcp_result(web_res))[:100]}...")

    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

def parse_mcp_result(result):
    """辅助函数：解析 MCP 工具返回的复杂结构"""
    if hasattr(result, 'content') and result.content:
        for item in result.content:
            if item.type == 'text':
                try:
                    return json.loads(item.text)
                except:
                    return item.text
    return str(result)

if __name__ == "__main__":
    asyncio.run(main())