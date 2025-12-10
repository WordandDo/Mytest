#!/usr/bin/env python3
"""
单资源分配流程快速测试

最小化测试脚本，验证：
1. allocate_single_resource 工具调用
2. setup_batch_resources 工具调用
3. release_batch_resources 工具调用

不依赖真实的 Resource Manager，使用模拟数据进行测试。
"""

import json
import logging
import asyncio
from typing import Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockMCPClient:
    """
    模拟 MCP 客户端，用于测试流程而无需真实的 MCP Server
    """

    def __init__(self):
        self.allocated_resources = {}

    async def connect(self):
        """模拟连接"""
        logger.info("✅ Mock: Connected to MCP Server")
        await asyncio.sleep(0.1)

    async def list_tools(self):
        """模拟列出工具"""
        tools = [
            {"name": "allocate_single_resource", "description": "Allocate a single resource"},
            {"name": "setup_batch_resources", "description": "Setup batch resources"},
            {"name": "get_batch_initial_observations", "description": "Get initial observations"},
            {"name": "release_batch_resources", "description": "Release batch resources"},
            {"name": "computer", "description": "Computer control tool"},
        ]
        logger.info(f"✅ Mock: Found {len(tools)} tools")
        return tools

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]):
        """模拟工具调用"""
        logger.info(f"🔧 Mock: Calling tool '{tool_name}' with args: {arguments}")
        await asyncio.sleep(0.2)  # 模拟网络延迟

        if tool_name == "allocate_single_resource":
            return self._mock_allocate_single(arguments)
        elif tool_name == "setup_batch_resources":
            return self._mock_setup_batch(arguments)
        elif tool_name == "get_batch_initial_observations":
            return self._mock_get_observations(arguments)
        elif tool_name == "release_batch_resources":
            return self._mock_release_batch(arguments)
        elif tool_name == "computer":
            return self._mock_computer_action(arguments)
        else:
            return self._mock_generic_response()

    def _mock_allocate_single(self, args):
        """模拟单资源分配"""
        resource_type = args.get("resource_type", "vm_pyautogui")
        worker_id = args.get("worker_id", "unknown")

        # 生成模拟资源 ID
        resource_id = f"{resource_type}_{worker_id}_001"

        # 构造返回数据
        result = {
            resource_type: {
                "id": resource_id,
                "ip": "192.168.1.100",
                "port": 5000,
                "token": "mock_auth_token_123"
            }
        }

        # 保存已分配资源
        self.allocated_resources[resource_id] = result[resource_type]

        logger.info(f"✅ Mock: Allocated resource '{resource_id}'")

        return self._wrap_response(result)

    def _mock_setup_batch(self, args):
        """模拟批量资源设置"""
        allocated_resources = args.get("allocated_resources", {})
        init_configs = args.get("resource_init_configs", {})

        details = {}
        for res_type in allocated_resources.keys():
            details[res_type] = {
                "success": True,
                "message": f"{res_type} Init Success (Mock)"
            }

        result = {
            "status": "success",
            "details": details
        }

        logger.info(f"✅ Mock: Setup completed for {len(allocated_resources)} resources")

        return self._wrap_response(result)

    def _mock_get_observations(self, args):
        """模拟获取初始观察"""
        worker_id = args.get("worker_id", "unknown")

        # 模拟观察数据
        observations = {}

        # 检查已分配的资源类型
        for resource_id, resource_info in self.allocated_resources.items():
            if "vm" in resource_id:
                resource_type = resource_id.split('_')[0] + '_' + resource_id.split('_')[1]
                observations[resource_type] = {
                    "screenshot": "mock_base64_screenshot_data_here...",
                    "accessibility_tree": "mock_accessibility_tree_data...",
                    "message": "Mock observation fetched from local controller"
                }
            elif "rag" in resource_id:
                observations["rag"] = {
                    "status": "ready",
                    "message": "Mock RAG session active"
                }

        logger.info(f"✅ Mock: Retrieved observations for {len(observations)} resources")

        return self._wrap_response(observations)

    def _mock_release_batch(self, args):
        """模拟批量资源释放"""
        resource_ids = args.get("resource_ids", [])
        worker_id = args.get("worker_id", "unknown")

        details = {}
        for rid in resource_ids:
            if rid in self.allocated_resources:
                del self.allocated_resources[rid]
                details[rid] = "released"
                logger.info(f"✅ Mock: Released resource '{rid}'")
            else:
                details[rid] = "not_found"
                logger.warning(f"⚠️ Mock: Resource '{rid}' not found")

        result = {
            "status": "completed",
            "details": details
        }

        return self._wrap_response(result)

    def _mock_computer_action(self, args):
        """模拟计算机控制动作"""
        action = args.get("action", "unknown")

        result = {
            "action": action,
            "status": "success",
            "message": f"Mock: {action} executed successfully"
        }

        logger.info(f"✅ Mock: Computer action '{action}' executed")

        return self._wrap_response(result)

    def _mock_generic_response(self):
        """通用模拟响应"""
        return self._wrap_response({"status": "success", "message": "Mock tool executed"})

    def _wrap_response(self, data: Dict[str, Any]):
        """包装响应为 MCP CallToolResult 格式"""
        class MockContent:
            def __init__(self, text):
                self.type = "text"
                self.text = text

        class MockResult:
            def __init__(self, content):
                self.content = content

        json_text = json.dumps(data)
        return MockResult([MockContent(json_text)])


class SimpleSingleResourceTest:
    """
    简化的单资源测试类
    """

    def __init__(self, use_mock: bool = True):
        self.worker_id = "test_worker_001"
        self.use_mock = use_mock

        if use_mock:
            self.client = MockMCPClient()
            logger.info("Using MockMCPClient for testing")
        else:
            # 实际生产环境使用真实客户端
            from utils.mcp_sse_client import MCPSSEClient
            self.client = MCPSSEClient("http://localhost:8080/sse")
            logger.info("Using real MCPSSEClient")

        self.allocated_resource_id = None

    async def run_test(self, resource_type: str = "vm_pyautogui"):
        """
        运行完整测试流程
        """
        logger.info("=" * 60)
        logger.info("开始单资源分配流程测试")
        logger.info("=" * 60)

        try:
            # 1. 连接
            logger.info("\n[步骤 1/6] 连接到 MCP Server")
            await self.client.connect()

            # 2. 列出工具
            logger.info("\n[步骤 2/6] 列出可用工具")
            tools = await self.client.list_tools()

            # 3. 分配资源
            logger.info(f"\n[步骤 3/6] 分配单资源: {resource_type}")
            alloc_result = await self.client.call_tool(
                "allocate_single_resource",
                {
                    "worker_id": self.worker_id,
                    "resource_type": resource_type,
                    "timeout": 600
                }
            )

            alloc_data = self._parse_response(alloc_result)
            logger.info(f"   分配结果: {json.dumps(alloc_data, indent=2)}")

            # 提取资源 ID
            if resource_type in alloc_data:
                self.allocated_resource_id = alloc_data[resource_type].get("id")
                logger.info(f"   ✅ 资源 ID: {self.allocated_resource_id}")
            else:
                raise ValueError("分配失败：未返回资源信息")

            # 4. 初始化资源
            logger.info("\n[步骤 4/6] 初始化资源")
            setup_result = await self.client.call_tool(
                "setup_batch_resources",
                {
                    "worker_id": self.worker_id,
                    "resource_init_configs": {},
                    "allocated_resources": alloc_data
                }
            )

            setup_data = self._parse_response(setup_result)
            logger.info(f"   初始化结果: {json.dumps(setup_data, indent=2)}")

            # 5. 获取初始观察
            logger.info("\n[步骤 5/6] 获取初始观察")
            obs_result = await self.client.call_tool(
                "get_batch_initial_observations",
                {
                    "worker_id": self.worker_id
                }
            )

            obs_data = self._parse_response(obs_result)
            # 简化观察数据显示（避免打印长 base64）
            simplified_obs = self._simplify_observation(obs_data)
            logger.info(f"   观察数据: {json.dumps(simplified_obs, indent=2)}")

            # 6. 释放资源
            logger.info("\n[步骤 6/6] 释放资源")
            release_result = await self.client.call_tool(
                "release_batch_resources",
                {
                    "worker_id": self.worker_id,
                    "resource_ids": [self.allocated_resource_id]
                }
            )

            release_data = self._parse_response(release_result)
            logger.info(f"   释放结果: {json.dumps(release_data, indent=2)}")

            logger.info("\n" + "=" * 60)
            logger.info("✅ 测试完成！所有步骤执行成功")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"\n❌ 测试失败: {e}", exc_info=True)
            raise

    def _parse_response(self, response):
        """解析 MCP 响应"""
        try:
            if hasattr(response, 'content') and response.content:
                content_item = response.content[0]
                if hasattr(content_item, 'text'):
                    return json.loads(content_item.text)
            return {}
        except Exception as e:
            logger.error(f"解析响应失败: {e}")
            return {}

    def _simplify_observation(self, obs_data: Dict[str, Any]) -> Dict[str, Any]:
        """简化观察数据以便显示"""
        simplified = {}
        for res_type, obs_content in obs_data.items():
            if isinstance(obs_content, dict):
                simplified[res_type] = {}
                for key, value in obs_content.items():
                    if key == "screenshot" and value:
                        simplified[res_type][key] = f"[Base64 Image, length={len(value)}]"
                    elif key == "accessibility_tree" and value:
                        simplified[res_type][key] = f"[Tree Data, length={len(value)}]"
                    else:
                        simplified[res_type][key] = value
            else:
                simplified[res_type] = obs_content
        return simplified


async def main():
    """
    主测试函数
    """
    import sys

    # 检查是否使用真实客户端
    use_mock = "--real" not in sys.argv

    logger.info(f"测试模式: {'Mock (模拟)' if use_mock else 'Real (真实)'}")

    # 资源类型
    resource_type = "vm_pyautogui"
    if len(sys.argv) > 1 and sys.argv[-1].startswith("vm_"):
        resource_type = sys.argv[-1]

    logger.info(f"资源类型: {resource_type}")

    # 运行测试
    test = SimpleSingleResourceTest(use_mock=use_mock)
    await test.run_test(resource_type=resource_type)


if __name__ == "__main__":
    """
    运行测试

    使用方法：
    1. 使用 Mock 客户端（不需要真实服务）：
       python test_single_resource_flow.py

    2. 使用真实 MCP 客户端：
       python test_single_resource_flow.py --real

    3. 指定资源类型：
       python test_single_resource_flow.py vm_computer_13
       python test_single_resource_flow.py --real vm_pyautogui
    """
    asyncio.run(main())
