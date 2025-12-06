import asyncio
import os
import sys
import shutil
from pathlib import Path

# 1. 确保项目根目录在 sys.path 中
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

try:
    from src.utils.search_v2 import CloudStorageService
    from src.utils.search_v2.config.settings import Config
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

def create_dummy_image(path: Path):
    """创建一个简单的红色 PNG 图片用于测试"""
    # 简单的 1x1 红色像素 PNG 文件的十六进制表示
    hex_data = (
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde'
        b'\x00\x00\x00\x0cIDAT\x08\xd7c\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82'
    )
    with open(path, 'wb') as f:
        f.write(hex_data)
    print(f"🖼️  已生成临时测试图片: {path}")

async def test_real_upload():
    print("\n" + "="*50)
    print("☁️  测试 123Pan 真实上传功能")
    print("="*50)

    # 检查 Access Token
    config = Config()
    token_path = Path(config.PAN123_ACCESS_TOKEN_FILE)
    
    if not token_path.exists():
        print(f"❌ 错误: 未找到 Access Token 文件")
        print(f"   请在以下路径创建文件并填入 Token: {token_path}")
        return

    # 创建服务实例
    try:
        service = CloudStorageService()
        print("✅ 服务初始化成功")
    except Exception as e:
        print(f"❌ 服务初始化失败: {e}")
        return

    # 准备测试文件
    test_file = project_root / "test_upload_image.png"
    create_dummy_image(test_file)

    try:
        print(f"🚀 开始上传文件: {test_file.name} ...")
        print(f"   目标文件夹 ID: {config.PAN123_PARENT_FILE_ID}")
        
        # 执行上传
        result = await service.upload_single_image(test_file)
        
        print("\n✅ 上传成功!")
        print(f"📄 文件名: {result.get('name')}")
        print(f"🆔 FileID: {result.get('fileID')}")
        print(f"🔗 URL: {result.get('url')}")
        
    except Exception as e:
        print(f"\n❌ 上传失败: {e}")
        print("提示: 请检查 Token 是否过期，或 PAN123_PARENT_FILE_ID 是否正确。")
    finally:
        # 清理临时文件
        if test_file.exists():
            os.remove(test_file)
            print(f"🧹 已清理临时文件")

if __name__ == "__main__":
    # 加载环境变量 (如果需要)
    from dotenv import load_dotenv
    load_dotenv()
    
    asyncio.run(test_real_upload())