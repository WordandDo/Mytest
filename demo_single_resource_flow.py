#!/usr/bin/env python3
"""
单资源分配与释放流程演示

基于 HttpMCPEnv 框架，展示如何使用 system_tools.py 中的单资源分配工具：
- allocate_single_resource: 单资源分配
- release_batch_resources: 批量资源释放（可用于单资源）

流程包含：
1. 环境初始化
2. 连接 MCP Server
3. 单资源分配（例如：vm_pyautogui）
4. 工具调用演示
5. 资源释放
6. 环境清理
"""

import os
import sys
import json
import logging
import asyncio

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.mcp_sse_client import MCPSSEClient

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleSingleResourceDemo:
    """
    简易单资源分配演示类

    展示如何使用 MCP 协议进行单资源的完整生命周期管理
    """

    def __init__(self, server_url: str = "http://localhost:8080"):
        """
        初始化演示环境

        Args:
            server_url: MCP Gateway Server 地址
        """
        self.server_url = server_url
        self.worker_id = "demo_worker_001"
        self.mcp_client = MCPSSEClient(f"{server_url}/sse")
        self.allocated_resource_id = None
        self.resource_type = None

        logger.info(f"[{self.worker_id}] SimpleSingleResourceDemo initialized")
        logger.info(f"[{self.worker_id}] Server URL: {server_url}")

    async def connect(self):
        """连接到 MCP Server"""
        logger.info(f"[{self.worker_id}] Connecting to MCP Server...")
        await self.mcp_client.connect()
        logger.info(f"[{self.worker_id}] ✅ Connected to MCP Server")

    async def list_available_tools(self):
        """列出所有可用工具"""
        logger.info(f"[{self.worker_id}] Fetching available tools...")
        tools = await self.mcp_client.list_tools()

        logger.info(f"[{self.worker_id}] Found {len(tools)} tools:")
        for tool in tools:
            logger.info(f"  - {tool.name}: {tool.description}")

        return tools

    async def allocate_single_resource(self, resource_type: str, timeout: int = 600):
        """
        分配单个资源

        Args:
            resource_type: 资源类型（如 'vm_pyautogui', 'vm_computer_13'）
            timeout: 分配超时时间（秒）

        Returns:
            bool: 分配是否成功
        """
        self.resource_type = resource_type
        logger.info(f"[{self.worker_id}] Allocating single resource: {resource_type}")

        try:
            # 调用 allocate_single_resource 工具
            result = await self.mcp_client.call_tool(
                "allocate_single_resource",
                {
                    "worker_id": self.worker_id,
                    "resource_type": resource_type,
                    "timeout": timeout
                }
            )

            # 解析结果
            response_data = self._parse_mcp_response(result)

            if response_data.get("status") == "error":
                logger.error(f"[{self.worker_id}] ❌ Allocation failed: {response_data.get('message')}")
                return False

            # 提取资源信息
            # 单资源分配返回格式：{"resource_type": {"id": "...", "ip": "...", ...}}
            resource_info = response_data.get(resource_type)
            if resource_info and isinstance(resource_info, dict):
                self.allocated_resource_id = resource_info.get("id")
                logger.info(f"[{self.worker_id}] ✅ Resource allocated successfully")
                logger.info(f"[{self.worker_id}]   Resource ID: {self.allocated_resource_id}")
                logger.info(f"[{self.worker_id}]   Resource Info: {json.dumps(resource_info, indent=2)}")
                return True
            else:
                logger.error(f"[{self.worker_id}] ❌ Invalid allocation response format")
                return False

        except Exception as e:
            logger.error(f"[{self.worker_id}] ❌ Allocation exception: {e}")
            return False

    async def setup_resource(self, allocated_data: dict, init_config: dict = None):
        """
        初始化已分配的资源

        Args:
            allocated_data: allocate_single_resource 返回的数据
            init_config: 资源初始化配置（可选）

        Returns:
            bool: 初始化是否成功
        """
        logger.info(f"[{self.worker_id}] Setting up resource...")

        try:
            # 调用 setup_batch_resources 工具
            # 即使只有一个资源，也使用批量接口（内部会自动同步会话）
            result = await self.mcp_client.call_tool(
                "setup_batch_resources",
                {
                    "worker_id": self.worker_id,
                    "resource_init_configs": init_config or {},
                    "allocated_resources": allocated_data
                }
            )

            # 解析结果
            response_data = self._parse_mcp_response(result)

            if response_data.get("status") in ["success", "partial_error"]:
                logger.info(f"[{self.worker_id}] ✅ Resource setup completed")
                logger.info(f"[{self.worker_id}]   Details: {json.dumps(response_data.get('details', {}), indent=2)}")
                return True
            else:
                logger.error(f"[{self.worker_id}] ❌ Setup failed: {response_data}")
                return False

        except Exception as e:
            logger.error(f"[{self.worker_id}] ❌ Setup exception: {e}")
            return False

    async def get_initial_observation(self):
        """
        获取资源的初始观察数据

        Returns:
            dict: 初始观察数据
        """
        logger.info(f"[{self.worker_id}] Fetching initial observation...")

        try:
            result = await self.mcp_client.call_tool(
                "get_batch_initial_observations",
                {
                    "worker_id": self.worker_id
                }
            )

            observation_data = self._parse_mcp_response(result)

            logger.info(f"[{self.worker_id}] ✅ Initial observation retrieved")

            # 显示观察数据的摘要（避免打印太长的 base64 图像）
            for res_type, obs_content in observation_data.items():
                if isinstance(obs_content, dict):
                    has_screenshot = "screenshot" in obs_content and obs_content["screenshot"]
                    has_a11y = "accessibility_tree" in obs_content and obs_content["accessibility_tree"]
                    logger.info(f"[{self.worker_id}]   {res_type}:")
                    logger.info(f"[{self.worker_id}]     - Screenshot: {'Yes' if has_screenshot else 'No'}")
                    logger.info(f"[{self.worker_id}]     - Accessibility Tree: {'Yes' if has_a11y else 'No'}")

            return observation_data

        except Exception as e:
            logger.error(f"[{self.worker_id}] ❌ Failed to get observation: {e}")
            return {}

    async def call_tool_example(self, tool_name: str, tool_args: dict):
        """
        调用工具示例（通用工具调用接口）

        Args:
            tool_name: 工具名称
            tool_args: 工具参数

        Returns:
            调用结果
        """
        logger.info(f"[{self.worker_id}] Calling tool: {tool_name}")
        logger.info(f"[{self.worker_id}]   Args: {json.dumps(tool_args, indent=2)}")

        try:
            # 自动注入 worker_id（如果工具需要）
            if "worker_id" not in tool_args:
                tool_args["worker_id"] = self.worker_id

            result = await self.mcp_client.call_tool(tool_name, tool_args)

            logger.info(f"[{self.worker_id}] ✅ Tool call completed")

            # 解析并显示结果
            if hasattr(result, 'content') and result.content:
                for item in result.content:
                    if item.type == 'text':
                        logger.info(f"[{self.worker_id}]   Result: {item.text[:200]}...")
                    elif item.type == 'image':
                        logger.info(f"[{self.worker_id}]   Result: [Image data]")

            return result

        except Exception as e:
            logger.error(f"[{self.worker_id}] ❌ Tool call failed: {e}")
            return None

    async def release_resource(self):
        """释放已分配的资源"""
        if not self.allocated_resource_id:
            logger.warning(f"[{self.worker_id}] No resource to release")
            return

        logger.info(f"[{self.worker_id}] Releasing resource: {self.allocated_resource_id}")

        try:
            # 调用 release_batch_resources 工具
            result = await self.mcp_client.call_tool(
                "release_batch_resources",
                {
                    "worker_id": self.worker_id,
                    "resource_ids": [self.allocated_resource_id]
                }
            )

            response_data = self._parse_mcp_response(result)

            if response_data.get("status") == "completed":
                logger.info(f"[{self.worker_id}] ✅ Resource released successfully")
                logger.info(f"[{self.worker_id}]   Details: {json.dumps(response_data.get('details', {}), indent=2)}")
                self.allocated_resource_id = None
            else:
                logger.error(f"[{self.worker_id}] ❌ Release failed: {response_data}")

        except Exception as e:
            logger.error(f"[{self.worker_id}] ❌ Release exception: {e}")

    async def disconnect(self):
        """断开与 MCP Server 的连接"""
        logger.info(f"[{self.worker_id}] Disconnecting from MCP Server...")
        # MCPSSEClient 可能需要实现 disconnect 方法
        # 这里暂时只记录日志
        logger.info(f"[{self.worker_id}] ✅ Disconnected")

    def _parse_mcp_response(self, response):
        """
        解析 MCP 响应

        Args:
            response: MCP CallToolResult

        Returns:
            dict: 解析后的数据
        """
        try:
            if hasattr(response, 'content') and response.content and len(response.content) > 0:
                content_item = response.content[0]
                if hasattr(content_item, 'text') and content_item.text:
                    return json.loads(content_item.text)
            return {"status": "unknown", "message": "Empty response"}
        except Exception as e:
            logger.error(f"Failed to parse MCP response: {e}")
            return {"status": "error", "message": str(e)}


async def main():
    """
    主流程演示

    演示步骤：
    1. 初始化环境
    2. 连接 MCP Server
    3. 列出可用工具
    4. 分配单个资源（vm_pyautogui）
    5. 初始化资源
    6. 获取初始观察
    7. 调用工具示例（如果适用）
    8. 释放资源
    9. 断开连接
    """

    # 从环境变量获取配置
    server_url = os.environ.get("MCP_SERVER_URL", "http://localhost:8080")
    resource_type = os.environ.get("DEMO_RESOURCE_TYPE", "vm_pyautogui")

    demo = SimpleSingleResourceDemo(server_url=server_url)

    try:
        # 步骤 1: 连接到 MCP Server
        await demo.connect()

        # 步骤 2: 列出可用工具
        tools = await demo.list_available_tools()

        # 步骤 3: 分配单个资源
        allocation_success = await demo.allocate_single_resource(resource_type)

        if not allocation_success:
            logger.error("❌ Resource allocation failed, exiting demo")
            return

        # 获取分配结果（模拟从 allocate_single_resource 返回的完整数据）
        # 注意：实际使用中应该从 allocation 调用返回值中提取
        # 这里为了演示，我们构造一个示例数据结构
        allocated_data = {
            resource_type: {
                "id": demo.allocated_resource_id,
                # 其他信息会在实际分配时返回
            }
        }

        # 步骤 4: 初始化资源
        setup_success = await demo.setup_resource(allocated_data)

        if not setup_success:
            logger.error("❌ Resource setup failed")
            await demo.release_resource()
            return

        # 步骤 5: 获取初始观察
        observation = await demo.get_initial_observation()

        # 步骤 6: 工具调用示例
        # 根据资源类型调用相应的工具
        if resource_type.startswith("vm_"):
            logger.info(f"[{demo.worker_id}] === Tool Call Example ===")
            # 示例：获取屏幕截图
            # await demo.call_tool_example(
            #     "computer",
            #     {
            #         "action": "screenshot"
            #     }
            # )
            logger.info(f"[{demo.worker_id}] (Tool call example skipped - add specific tool calls here)")

        # 步骤 7: 释放资源
        await demo.release_resource()

        # 步骤 8: 断开连接
        await demo.disconnect()

        logger.info("✅ Demo completed successfully!")

    except Exception as e:
        logger.error(f"❌ Demo failed with exception: {e}", exc_info=True)

        # 确保资源被释放
        if demo.allocated_resource_id:
            await demo.release_resource()


if __name__ == "__main__":
    """
    运行演示

    环境变量配置：
    - MCP_SERVER_URL: MCP Gateway Server 地址（默认: http://localhost:8080）
    - DEMO_RESOURCE_TYPE: 要分配的资源类型（默认: vm_pyautogui）
    - RESOURCE_API_URL: Resource API 地址（默认: http://localhost:8000）

    使用示例：
    ```bash
    # 使用默认配置
    python demo_single_resource_flow.py

    # 自定义配置
    MCP_SERVER_URL=http://localhost:8080 \
    DEMO_RESOURCE_TYPE=vm_computer_13 \
    RESOURCE_API_URL=http://localhost:8000 \
    python demo_single_resource_flow.py
    ```
    """

    # 设置默认环境变量（如果未设置）
    if "RESOURCE_API_URL" not in os.environ:
        os.environ["RESOURCE_API_URL"] = "http://localhost:8000"

    # 运行异步主函数
    asyncio.run(main())
