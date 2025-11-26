import logging
import json
from typing import Dict, Any, List
from utils.mcp_client import MCPClientWrapper

logger = logging.getLogger(__name__)

class OSWorldMCPEnv:
    """
    Native MCP Environment.
    不再继承旧的 Environment 类，抛弃所有历史包袱。
    """
    def __init__(self, server_script: str, env_id: str):
        self.client = MCPClientWrapper(server_script)
        self.env_id = env_id # 用于区分不同的并发任务
        self.tools_schema = []

    async def connect(self):
        """建立连接并同步工具定义"""
        await self.client.connect()
        
        # 1. 动态获取 Server 端的工具列表
        tools_list = await self.client.session.list_tools()
        
        # 2. 转换为 OpenAI 格式
        self.tools_schema = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            } 
            for tool in tools_list.tools
        ]
        logger.info(f"[{self.env_id}] Connected. Discovered {len(self.tools_schema)} tools.")

    async def setup_task(self, config_name: str, task_id: str) -> Dict[str, Any]:
        """直接调用 Server 的 setup_environment"""
        # 注意：我们在 Server 端实现 setup 时，应该让它返回初始的 observation
        result = await self.client.call_tool("setup_environment", {
            "config_name": config_name,
            "task_id": task_id
        })
        
        # 解析返回结果 (假设 Server 返回的是 JSON 字符串)
        if isinstance(result, str):
            try:
                return json.loads(result)
            except:
                return {"status": "error", "raw": result}
        return result

    async def step(self, tool_name: str, args: Dict) -> Dict[str, Any]:
        """执行动作"""
        # 直接转发，不做任何处理
        raw_result = await self.client.call_tool(tool_name, args)
        
        # 统一回包格式
        if isinstance(raw_result, str):
            try:
                return json.loads(raw_result)
            except:
                return {"text": raw_result} # Fallback for plain text
        return raw_result

    async def evaluate(self) -> float:
        """
        [重大变更] 评估逻辑移至 Server 端。
        不再在 Client 跑 evaluators，而是发送指令让 Server 检查状态。
        """
        try:
            res = await self.client.call_tool("evaluate_task", {})
            return float(res)
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return 0.0

    async def teardown(self):
        """资源释放"""
        await self.client.call_tool("teardown_environment", {})
        await self.client.close()