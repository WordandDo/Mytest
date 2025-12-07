# debug_rag_server.py
import sys
import os
import json
import logging

# 添加 src 路径
sys.path.append(os.path.join(os.getcwd(), "src"))

from utils.resource_pools.rag_pool import start_rag_server

def debug_main():
    # 1. 读取配置文件
    config_path = "deployment_config.json"
    if not os.path.exists(config_path):
        print(f"❌ 错误: 找不到配置文件 {config_path}")
        return

    print(f"📖 读取配置文件: {config_path}...")
    with open(config_path, "r") as f:
        deploy_config = json.load(f)

    # 2. 提取 RAG 配置
    # 根据你的配置，这里可能是 rag_hybrid 或 rag
    rag_config = deploy_config.get("resources", {}).get("rag_hybrid", {}).get("config", {})
    
    if not rag_config:
        print("⚠️ 未找到 rag_hybrid 配置，尝试查找 rag 配置...")
        rag_config = deploy_config.get("resources", {}).get("rag", {}).get("config", {})

    if not rag_config:
        print("❌ 错误: 配置文件中没有找到有效的 RAG 配置")
        return

    print("\n⚙️  RAG 服务配置:")
    print(json.dumps(rag_config, indent=2, ensure_ascii=False))
    print("-" * 60)
    print("🚀 正在尝试独立启动 RAG Server (端口 8001)...")
    print("⚠️  请注意观察下方的报错信息 (Traceback)")
    print("-" * 60)

    # 3. 启动服务 (这会阻塞当前窗口，直到出错或被 Ctrl+C 中断)
    try:
        start_rag_server(8001, rag_config)
    except SystemExit as e:
        print(f"\n❌ RAG Server 进程退出，代码: {e}")
    except Exception as e:
        print(f"\n❌ 发生未捕获异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_main()