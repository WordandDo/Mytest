import logging
from typing import List, Callable, Dict

# --- 导入具体的工具函数 ---
# 1. 导入 RAG 相关的函数 (来自之前修改的 rag_server.py)
try:
    from mcp_server.rag_server import (
        setup_rag_engine, 
        query_knowledge_base, 
        release_rag_engine
    )
except ImportError as e:
    logging.warning(f"Failed to import RAG functions: {e}")
    # 定义伪函数以防报错
    async def setup_rag_engine(worker_id: str): pass
    async def query_knowledge_base(worker_id: str, query: str, top_k: int = 3): pass
    async def release_rag_engine(worker_id: str): pass

# 2. 导入 OSWorld/VM 相关的函数
# 假设这些函数定义在 src/mcp_server/osworld_server.py 或类似位置
# 如果这些函数目前是 @mcp.tool() 装饰的，直接导入函数名即可，装饰器不影响函数本身的引用
try:
    from mcp_server.osworld_server import (
        start_vm,
        stop_vm,
        get_screenshot,
        execute_action
    )
except ImportError:
    # 兼容性处理：如果文件不存在，定义伪函数以防报错
    async def start_vm(worker_id: str): pass
    async def stop_vm(worker_id: str): pass
    async def get_screenshot(worker_id: str): return "base64..."
    async def execute_action(worker_id: str, action: str): return "done"

logger = logging.getLogger("ToolRegistry")

class ToolRegistry:
    """
    工具注册中心。
    负责维护 '工具组名称' 到 '具体函数列表' 的映射。
    """
    
    _REGISTRY: Dict[str, List[Callable]] = {
        # === RAG 工具组 ===
        "rag_lifecycle": [
            setup_rag_engine,
            release_rag_engine
        ],
        "rag_query": [
            query_knowledge_base
        ],
        
        # === VM/Desktop 工具组 ===
        "computer_lifecycle": [
            start_vm,
            stop_vm
        ],
        "desktop_observation": [
            get_screenshot
        ],
        "desktop_action": [
            execute_action
        ]
    }

    @classmethod
    def get_tools_by_group(cls, group_name: str) -> List[Callable]:
        """根据组名获取工具函数列表"""
        tools = cls._REGISTRY.get(group_name)
        if not tools:
            logger.warning(f"Tool group '{group_name}' not found in registry.")
            return []
        return tools

    @classmethod
    def get_tools_by_config(cls, module_config: dict) -> List[Callable]:
        """
        根据配置项自动解析需要的工具。
        支持根据 resource_type 或 action_space 进行特殊处理。
        """
        tools = []
        
        # 1. 显式指定的工具组
        groups = module_config.get("tool_groups", [])
        for group in groups:
            tools.extend(cls.get_tools_by_group(group))
            
        # 2. 根据 resource_type 自动追加默认组 (可选逻辑)
        r_type = module_config.get("resource_type")
        if r_type == "rag" and "rag_lifecycle" not in groups:
             tools.extend(cls.get_tools_by_group("rag_lifecycle"))

        # 3. 去重 (保持顺序)
        seen = set()
        unique_tools = []
        for tool in tools:
            if tool not in seen:
                seen.add(tool)
                unique_tools.append(tool)
        
        return unique_tools