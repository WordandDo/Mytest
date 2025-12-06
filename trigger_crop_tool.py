import asyncio
import base64
import json
import io
import sys
import os

# 确保可以将 src 目录加入路径以导入项目模块
sys.path.append(os.getcwd())

try:
    from src.utils.mcp_sse_client import MCPSSEClient
except ImportError:
    print("❌ Error: 无法导入 MCPSSEClient。请确保您在项目根目录下运行此脚本。")
    sys.exit(1)

def create_dummy_base64_image(color=(255, 0, 0), size=(100, 100)):
    """
    生成一个指定颜色和大小的内存图片，并返回 Base64 字符串。
    需要安装 Pillow: pip install Pillow
    """
    try:
        from PIL import Image
    except ImportError:
        print("❌ Error: 需要安装 Pillow 库来生成测试图片 (pip install Pillow)")
        sys.exit(1)

    img = Image.new('RGB', size, color)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

async def main():
    # =========================================================================
    # 1. 准备测试数据
    # =========================================================================
    print("🎨 生成测试图片...")
    # 生成一张红色图片 (标记为 img_red) 和一张蓝色图片 (标记为 img_blue)
    b64_red = create_dummy_base64_image(color=(255, 0, 0), size=(200, 200))
    b64_blue = create_dummy_base64_image(color=(0, 0, 255), size=(200, 200))

    # 构造对话历史 (Messages)
    # ImageProcessor 的解析逻辑要求：Token 必须在 image_url 之前的 text 块中
    conversation_history = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": "这是第一张红色的图片，标记为 <img_red>"
                },
                {
                    "type": "image_url", 
                    "image_url": {"url": b64_red}
                },
                {
                    "type": "text", 
                    "text": "\n这是第二张蓝色的图片，标记为 <img_blue>"
                },
                {
                    "type": "image_url", 
                    "image_url": {"url": b64_blue}
                }
            ]
        }
    ]

    # =========================================================================
    # 2. 初始化 Client 并连接 Server
    # =========================================================================
    server_url = "http://localhost:8080"
    print(f"🔌 连接到 MCP Server: {server_url} ...")
    
    client = MCPSSEClient(f"{server_url}/sse")
    
    # 启动 Client (Context Manager 方式或手动 connect)
    # 这里我们手动 connect 以演示流程
    await client.connect()
    
    try:
        # =========================================================================
        # 3. 构造工具调用参数
        # =========================================================================
        tool_name = "crop_images_by_token"
        
        # 裁切配置：[left, top, right, bottom]
        # img_red: 裁切左上角 50x50
        # img_blue: 裁切中间区域
        crop_config = {
            "img_red": [0, 0, 50, 50],
            "img_blue": [50, 50, 150, 150]
        }

        arguments = {
            "crop_config": crop_config,
            # 显式传入 messages，模拟 Gateway 的上下文注入
            "messages": conversation_history,
            # 使用 local 模式方便在本地文件夹查看结果，cloud 模式会尝试上传
            "storage_mode": "local" 
        }

        print(f"🔨 调用工具: {tool_name}")
        print(f"   配置: {json.dumps(crop_config)}")

        # =========================================================================
        # 4. 执行调用
        # =========================================================================
        # 注意：call_tool 内部会处理 JSON RPC
        result = await client.call_tool(tool_name, arguments)

        # =========================================================================
        # 5. 解析结果
        # =========================================================================
        print("\n✅ 工具调用完成! 结果如下:")
        
        # 解析 MCP 返回的 Content 对象
        if hasattr(result, 'content') and result.content:
            for item in result.content:
                if item.type == 'text':
                    try:
                        # 工具返回的是 JSON 字符串，尝试解析以便美化打印
                        res_json = json.loads(item.text)
                        print(json.dumps(res_json, indent=2, ensure_ascii=False))
                    except:
                        print(item.text)
        else:
            print(result)

    except Exception as e:
        print(f"\n❌ 调用失败: {e}")
    
    finally:
        # 清理连接
        # 注意：MCPSSEClient 可能没有显式的 close 方法，取决于具体实现
        # 这里直接让脚本结束即可
        pass

if __name__ == "__main__":
    asyncio.run(main())