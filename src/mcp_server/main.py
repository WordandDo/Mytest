import argparse
import json
import logging
import os
import sys
from typing import List

# 确保 src 目录在路径中
cwd = os.getcwd()
sys.path.append(cwd)
if os.path.join(cwd, "src") not in sys.path:
    sys.path.append(os.path.join(cwd, "src"))

from mcp.server.fastmcp import FastMCP
from mcp_server.registry import ToolRegistry

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GatewayServer")

def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Unified MCP Gateway Server")
    parser.add_argument("--config", type=str, default="config.json", help="Path to the server configuration file")
    parser.add_argument("--port", type=int, help="Override port defined in config")
    args = parser.parse_args()

    # 1. 加载配置
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        # 如果没有配置文件，提供一个默认的回退配置或退出
        logger.warning("Using default empty config")
        config = {"server_name": "Default Gateway", "modules": []}

    server_name = config.get("server_name", "Unified Gateway")
    
    # 2. 初始化 FastMCP Server
    # 注意: dependencies 参数用于管理生命周期，如果工具需要共享状态，可以在这里传入
    mcp = FastMCP(server_name)
    
    logger.info(f"🚀 Initializing {server_name}...")

    # 3. 动态注册工具
    registered_tools_count = 0
    modules = config.get("modules", [])
    
    for module in modules:
        r_type = module.get("resource_type", "unknown")
        action_space = module.get("action_space", "default")
        
        logger.info(f"Loading module: Type={r_type}, ActionSpace={action_space}")
        
        # 通过注册表获取该模块对应的所有 Python 函数
        tool_functions = ToolRegistry.get_tools_by_config(module)
        
        for func in tool_functions:
            try:
                # 将函数注册为 MCP Tool
                # FastMCP 会自动解析函数的 type hints 和 docstrings 作为工具描述
                mcp.tool()(func)
                logger.info(f"  + Registered tool: {func.__name__}")
                registered_tools_count += 1
            except Exception as e:
                logger.error(f"  - Failed to register tool {func.__name__}: {e}")

    logger.info(f"✅ Total tools registered: {registered_tools_count}")

    # 4. 启动服务器
    port = args.port if args.port else config.get("port", 8080)
    host = config.get("host", "0.0.0.0")
    debug = config.get("debug", False)

    logger.info(f"Starting SSE server on {host}:{port}")
    
    # 使用 SSE 传输模式 (与 Claude Desktop 等客户端兼容性最好)
    mcp.run(transport='sse')

if __name__ == "__main__":
    main()