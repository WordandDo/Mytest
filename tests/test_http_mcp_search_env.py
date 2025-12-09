import unittest
import json
import sys
import os
import asyncio
from unittest.mock import MagicMock, patch, mock_open, AsyncMock

# 将 src 加入路径以便导入模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from envs.http_mcp_search_env import HttpMCPSearchEnv

# 模拟 MCP 工具对象
class MockMCPTool:
    def __init__(self, name, description, schema):
        self.name = name
        self.description = description
        self.inputSchema = schema

class TestHttpMCPSearchEnv(unittest.TestCase):
    
    def setUp(self):
        # 1. Mock MCPSSEClient
        self.mcp_client_patcher = patch('envs.http_mcp_env.MCPSSEClient')
        self.MockMCPSSEClient = self.mcp_client_patcher.start()
        
        # 配置 Mock Client 实例
        self.mock_client_instance = self.MockMCPSSEClient.return_value
        
        # [关键修正] 使用 AsyncMock 避免 asyncio 事件循环冲突
        self.mock_client_instance.connect = AsyncMock(return_value=True)
        self.mock_client_instance.call_tool = AsyncMock() # 确保 call_tool 也是异步的
        
        # 模拟 list_tools 返回
        async def mock_list_tools():
            return [
                MockMCPTool(
                    name="web_search", 
                    description="Search the web.",
                    schema={"type": "object", "properties": {"query": {"type": "string"}}}
                )
            ]
        self.mock_client_instance.list_tools = MagicMock(side_effect=mock_list_tools)

        # 2. Mock OpenAI
        self.openai_patcher = patch('envs.http_mcp_env.openai.OpenAI')
        self.MockOpenAI = self.openai_patcher.start()
        
    def tearDown(self):
        self.mcp_client_patcher.stop()
        self.openai_patcher.stop()

    def test_initialization_stateless(self):
        """测试环境初始化时是否正确设置为无状态模式"""
        mock_config = {
            "modules": [
                {"resource_type": "vm_pyautogui", "tool_groups": ["desktop"]},
                {"resource_type": "utility", "tool_groups": ["search_tools"]}
            ]
        }
        
        with patch("builtins.open", mock_open(read_data=json.dumps(mock_config))):
            with patch("os.path.exists", return_value=True):
                env = HttpMCPSearchEnv(worker_id="test_worker")
                self.addCleanup(env.env_close) # 确保清理 loop
                
                # 断言 active_resources 为空 (stateless mode)
                self.assertEqual(env.active_resources, [], "Search Env must be stateless (empty active_resources)")
                self.assertEqual(env.mode, "http_mcp_search")

    def test_config_filtering_logic(self):
        """测试 _load_gateway_config 是否正确过滤了非 utility 模块"""
        mock_config_content = json.dumps({
            "modules": [
                {"resource_type": "vm_computer", "tool_groups": ["basic"]}, # Should be removed
                {"resource_type": "utility", "tool_groups": ["search"]},    # Should be kept
                {"resource_type": "system", "tool_groups": ["sys"]}         # Should be removed by filtering logic
            ]
        })

        with patch("builtins.open", mock_open(read_data=mock_config_content)):
            with patch("os.path.exists", return_value=True):
                env = HttpMCPSearchEnv()
                self.addCleanup(env.env_close) # 确保清理 loop
                
                modules = env.modules_config["modules"]
                self.assertEqual(len(modules), 1, f"Should verify 1 module remains, got {len(modules)}")
                self.assertEqual(modules[0]["resource_type"], "utility")

    @patch('envs.http_mcp_env.HttpMCPEnv._call_tool_sync')
    def test_run_task_flow(self, mock_call_tool_sync):
        """测试完整的任务执行流程：提问 -> 调用工具 -> 得到答案"""
        
        # 准备环境配置
        mock_config = {"modules": [{"resource_type": "utility"}]}
        with patch("builtins.open", mock_open(read_data=json.dumps(mock_config))):
             with patch("os.path.exists", return_value=True):
                env = HttpMCPSearchEnv()
                self.addCleanup(env.env_close) # [修复 ResourceWarning]
                
                # 手动触发连接以加载工具
                env.env_start()

        # === [核心修复] 构造 Mock 消息，使 model_dump() 返回真实字典 ===
        
        # 1. 第一轮消息：工具调用
        tool_call_mock = MagicMock()
        tool_call_mock.id = "call_test_1"
        tool_call_mock.function.name = "web_search"
        tool_call_mock.function.arguments = '{"query": "MCP protocol"}'

        msg_round_1 = MagicMock()
        msg_round_1.role = "assistant"
        msg_round_1.content = None
        msg_round_1.tool_calls = [tool_call_mock]
        # 关键：model_dump 必须返回字典，以便后续逻辑可以读取 role/content
        msg_round_1.model_dump.return_value = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_test_1",
                "function": {
                    "name": "web_search",
                    "arguments": '{"query": "MCP protocol"}'
                }
            }]
        }

        # 2. 第二轮消息：最终答案
        msg_round_2 = MagicMock()
        msg_round_2.role = "assistant"
        msg_round_2.content = "MCP stands for Model Context Protocol."
        msg_round_2.tool_calls = None
        msg_round_2.model_dump.return_value = {
            "role": "assistant",
            "content": "MCP stands for Model Context Protocol."
        }

        # 配置 OpenAI Mock 返回
        mock_chat = self.MockOpenAI.return_value.chat.completions.create
        mock_chat.side_effect = [
            MagicMock(choices=[MagicMock(message=msg_round_1)]), # Round 1
            MagicMock(choices=[MagicMock(message=msg_round_2)])  # Round 2
        ]

        # 模拟工具执行结果
        mock_call_tool_sync.return_value = {"text": "Search Result: MCP is an open standard...", "images": []}

        # 运行任务
        task = {"id": "test_001", "question": "What is MCP?"}
        agent_config = {"model_name": "gpt-4-test", "max_turns": 3}
        logger = MagicMock()

        result = env.run_task(task, agent_config, logger)

        # 断言
        self.assertTrue(result["success"])
        self.assertEqual(result["answer"], "MCP stands for Model Context Protocol.")
        
        # 验证工具是否被正确调用
        mock_call_tool_sync.assert_called_with("web_search", {"query": "MCP protocol"})
        
        # 验证 System Prompt 是否注入了搜索专用指令
        call_args = mock_chat.call_args_list[0]
        messages = call_args[1]['messages']
        system_content = messages[0]['content']
        self.assertIn("You are a capable Search & Analysis Agent", system_content)

if __name__ == '__main__':
    unittest.main()