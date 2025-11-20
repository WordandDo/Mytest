# -*- coding: utf-8 -*-
"""
Environment Base Class (Simplified)

只保留核心架构：配置管理、工具注册、资源接口与任务入口。
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from utils.resource_manager import ResourceManager


class Tool(ABC):
    """工具抽象基类"""
    @property
    @abstractmethod
    def name(self) -> str: pass
    
    @property
    @abstractmethod
    def description(self) -> str: pass
    
    @property
    @abstractmethod
    def parameters(self) -> List[Dict[str, Any]]: pass
    
    @abstractmethod
    def call(self, params: Union[str, dict], **kwargs) -> str: pass


class Environment(ABC):
    """
    Agent 环境基类 (精简版)
    
    职责：
    1. 定义环境接口规范 (Mode, Run Task)
    2. 提供基础工具管理 (Register, Execute, Schema)
    3. 提供资源管理接口 (Setup Global Resources)
    """
    
    def __init__(self, 
                 model_name: str = "gpt-4.1-2025-04-14",
                 resource_manager: Optional['ResourceManager'] = None,
                 **kwargs):
        """
        基础初始化。
        
        Args:
            model_name: 默认模型名称
            resource_manager: 资源管理器实例 (由 setup_global_resources 创建)
            **kwargs: 其他配置参数 (保存至 self.config)
        """
        self.model_name = model_name
        self.config = kwargs
        
        # 工具管理
        self.tools: Dict[str, Tool] = {}
        self.tool_schemas: List[Dict[str, Any]] = []
        self.tool_descriptions: str = ""
        
        # 资源管理
        self._resource_manager = resource_manager
        if self._resource_manager is None:
            from utils.resource_manager import NoResourceManager
            self._resource_manager = NoResourceManager()
            
        # 自动调用子类工具初始化
        self._initialize_tools()

    # =========================================================================
    # 1. 核心抽象接口 (开发者必须实现)
    # =========================================================================
    
    @property
    @abstractmethod
    def mode(self) -> str:
        """返回环境模式名称 (如 'math', 'osworld')"""
        pass

    @abstractmethod
    def _initialize_tools(self):
        """在此方法中注册环境所需的工具"""
        pass

    @abstractmethod
    def run_task(self, task: Dict[str, Any], agent_config: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
        """
        核心任务执行入口。
        必须封装完整的 Agent 闭环：Prompt -> LLM -> Tool Execution -> Result
        """
        raise NotImplementedError("Subclasses must implement run_task")

    # =========================================================================
    # 2. 资源管理接口 (主进程调用)
    # =========================================================================

    @classmethod
    def setup_global_resources(cls, config: Any) -> 'ResourceManager':
        """
        类方法：初始化全局资源 (如 VM 池)。
        默认返回空管理器，需要重资产的环境(如 OSWorld)需重写此方法。
        """
        from utils.resource_manager import NoResourceManager
        return NoResourceManager()

    @property
    def resource_manager(self) -> 'ResourceManager':
        return self._resource_manager

    # =========================================================================
    # 3. 工具管理设施 (已实现，通常无需修改)
    # =========================================================================

    def register_tool(self, tool: Tool):
        """注册工具并自动更新 Schema"""
        self.tools[tool.name] = tool
        self._update_tool_metadata()

    def get_tool(self, name: str) -> Optional[Tool]:
        return self.tools.get(name)

    def execute_tool(self, tool_name: str, params: Union[str, dict], **kwargs) -> str:
        """执行工具的安全包装器"""
        tool = self.get_tool(tool_name)
        if not tool:
            return f"Error: Tool '{tool_name}' not found"
        try:
            return tool.call(params, **kwargs)
        except Exception as e:
            return f"Error executing '{tool_name}': {str(e)}"

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """获取用于 LLM API 的工具定义"""
        return self.tool_schemas

    def get_tool_descriptions(self) -> str:
        """获取用于 Prompt 的工具描述文本"""
        return self.tool_descriptions

    def _update_tool_metadata(self):
        """(内部) 生成工具 Schema 和描述"""
        self.tool_schemas = [self._tool_to_schema(t) for t in self.tools.values()]
        self.tool_descriptions = "\n".join([f"- {t.name}: {t.description}" for t in self.tools.values()])

    def _tool_to_schema(self, tool: Tool) -> Dict[str, Any]:
        """(内部) 将 Tool 转换为 OpenAI Schema 格式"""
        properties = {
            p['name']: {"type": p['type'], "description": p['description']}
            for p in tool.parameters
        }
        required = [p['name'] for p in tool.parameters if p.get('required', False)]
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }

    # =========================================================================
    # 4. 生命周期钩子 (可选覆盖)
    # =========================================================================

    def env_start(self) -> None:
        """Benchmark 开始时调用 (可选初始化)"""
        pass

    def env_close(self) -> None:
        """Benchmark 结束时调用 (可选清理)"""
        pass

    # =========================================================================
    # 5. 配置管理钩子 (可选覆盖，供子类调用 super())
    # =========================================================================

    def _initialize_config(self) -> None:
        """初始化配置 (可选覆盖)"""
        pass

    def _validate_config(self) -> None:
        """验证配置 (可选覆盖)"""
        pass
