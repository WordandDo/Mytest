# src/envs/http_mcp_env.py
import sys
import os
import json
import logging
import asyncio
import time
import random
from typing import Dict, Any, Union, Optional, List
from tools.tool import Tool
# 引入 MCP SDK
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.types import CallToolResult, TextContent

# 引入基类
from envs.enviroment import Environment

logger = logging.getLogger(__name__)

class HttpMCPEnv(Environment):
    """
    MCP SSE 环境适配器 (Client)
    
    特性：
    1. 使用标准 MCP 协议 (SSE + JSON-RPC) 与 Server 通信。
    2. 支持动态资源调度 (N-Worker M-VM 模型)，任务开始时申请资源，结束时释放。
    3. 自动从 Server 发现并注册工具。
    """
    
    # 开启重型资源模式，通知框架在 run_task 前后调用 allocate/release
    has_heavy_resource = True 

    def __init__(self, resource_manager=None, parallel_degree=1, **kwargs):
        # 1. 基础配置
        self.server_url = kwargs.get("mcp_server_url", "http://localhost:8080")
        self.resource_api_url = kwargs.get("resource_api_url", "http://localhost:8000")
        self.config_name = "default"
        
        # 2. 获取 worker_id (必需，作为会话标识)
        if "worker_id" in kwargs:
            self.worker_id = kwargs["worker_id"]
        else:
            import multiprocessing
            self.worker_id = multiprocessing.current_process().name
            
        logger.info(f"HttpMCPEnv initialized for {self.worker_id} -> {self.server_url} (SSE Mode)")
        
        # 3. 调用父类初始化 (会触发 _initialize_tools)
        super().__init__(**kwargs)

    @property
    def mode(self) -> str:
        return "http_mcp"

    # =========================================================================
    # 1. 工具发现与初始化
    # =========================================================================

    def _initialize_tools(self):
        """
        [重构版] 直接从 MCP 元数据生成 Schema 和 Description，
        跳过 self.tools 的对象注册过程。
        """
        try:
            logger.info(f"[{self.worker_id}] Fetching tools from MCP Server...")
            mcp_tools = self._list_tools_sync()
            
            # 1. 过滤黑名单工具
            blacklist = {
                "setup_environment", 
                "teardown_environment", 
                "get_observation", 
                "evaluate_task"
            }
            valid_tools = [t for t in mcp_tools if t.name not in blacklist]
            
            # 2. 直接生成 OpenAI Schema (用于 API 调用)
            # 这一步你原本就有了，保持不变
            self.tool_schemas = [self._convert_mcp_tool_to_openai(t) for t in valid_tools]
            
            # 3. [关键修改] 直接生成工具描述文本 (用于 System Prompt)
            # 基类原本是通过遍历 self.tools 生成的，这里我们直接手动生成
            descriptions = []
            for t in valid_tools:
                # 处理可能为空的描述
                desc = t.description if t.description else "No description provided."
                descriptions.append(f"- {t.name}: {desc}")
            
            self.tool_descriptions = "\n".join(descriptions)
            
            # 4. [关键修改] 不再操作 self.tools
            # self.tools = {}  # 保持为空即可，因为我们已经覆盖了所有依赖它的下游产物
                
            logger.info(f"[{self.worker_id}] Initialized {len(valid_tools)} tools (Metadata only).")
            
        except Exception as e:
            logger.error(f"Failed to initialize tools: {e}")
            self.tool_schemas = []
            self.tool_descriptions = "Error loading tools."
    def _convert_mcp_tool_to_openai(self, mcp_tool) -> Dict[str, Any]:
        """将 MCP Tool 对象转换为 OpenAI Function Schema"""
        
        # 1. 获取原始参数定义
        parameters = mcp_tool.inputSchema.copy() if hasattr(mcp_tool, "inputSchema") else {}
        
        # 2. [新增] 从 properties 中移除 worker_id
        if "properties" in parameters and "worker_id" in parameters["properties"]:
            del parameters["properties"]["worker_id"]
            
        # 3. [新增] 从 required 中移除 worker_id
        if "required" in parameters and "worker_id" in parameters["required"]:
            parameters["required"] = [p for p in parameters["required"] if p != "worker_id"]

        return {
            "type": "function",
            "function": {
                "name": mcp_tool.name,
                "description": mcp_tool.description,
                "parameters": parameters 
            }
        }
    def get_tool(self, name: str) -> Optional[Tool]:
        """
        [重写] 明确禁止获取本地工具实例
        """
        # 检查工具是否在我们的 Schema 列表中（确认它是否存在）
        tool_exists = any(s["function"]["name"] == name for s in self.tool_schemas)
        
        if tool_exists:
            # 记录警告日志，或者直接抛出错误
            logger.warning(
                f"Attempted to access local tool instance '{name}', but this is a remote MCP environment. "
                f"Use execute_tool('{name}', ...) instead."
            )
            return None # 或者 raise NotImplementedError(...)
            
        return None
    # 为了保持一致性，建议重写 list_tools，因为 self.tools 现在是空的
    def list_tools(self) -> List[str]:
        """列出当前可用工具名称 (从 Schema 中获取)"""
        return [s["function"]["name"] for s in self.tool_schemas]
    # =========================================================================
    # 2. 资源调度接口 (Dynamic Scheduling)
    # =========================================================================

    def env_start(self):
        """
        Worker 启动时的钩子。
        在动态调度模式下，此处不申请资源，仅做日志记录。
        """
        logger.info(f"Worker [{self.worker_id}] started (Dynamic Scheduling Mode)")

    def allocate_resource(self, worker_id: str) -> bool:
        """
        任务开始前调用：申请资源 (VM)
        实现循环等待逻辑，处理资源池满的情况。
        """
        logger.info(f"Worker [{worker_id}] acquiring VM for new task...")
        retry_interval = 5
        max_retries = 100 # 防止死循环，可根据 timeout 设置
        
        for attempt in range(max_retries):
            try:
                # 调用 MCP 的 setup_environment
                # 注意：Server 端需透传调用 Resource API
                res = self._call_tool_sync("setup_environment", {
                    "config_name": self.config_name,
                    "task_id": "dynamic_alloc",
                    "worker_id": worker_id 
                })
                
                # 检查结果
                # 如果返回的是 JSON 字符串，先解析
                if isinstance(res, str):
                    try: res_json = json.loads(res)
                    except: res_json = {"raw": res}
                else:
                    res_json = res if isinstance(res, dict) else {}

                if res_json.get("status") == "success":
                    logger.info(f"Worker [{worker_id}] acquired VM successfully.")
                    return True
                
                # 检查是否资源忙
                msg = res_json.get("message", "")
                if "No resources" in msg or "exhausted" in msg or "503" in msg:
                    if attempt % 2 == 0: # 减少日志频率
                        logger.info(f"Pool full, Worker [{worker_id}] waiting...")
                    time.sleep(retry_interval)
                    continue
                
                # 其他错误 (如网络错误)
                logger.error(f"Allocation error: {res}")
                time.sleep(retry_interval)
                
            except Exception as e:
                logger.error(f"Allocation exception: {e}")
                time.sleep(retry_interval)
        
        logger.error(f"Worker [{worker_id}] failed to acquire resource after retries.")
        return False

    def release_resource(self, worker_id: str, reset: bool = True) -> None:
        """
        任务结束后调用：释放资源
        """
        logger.info(f"Worker [{worker_id}] releasing VM...")
        try:
            self._call_tool_sync("teardown_environment", {"worker_id": worker_id})
        except Exception as e:
            logger.warning(f"Release failed: {e}")

    def get_allocated_resource_id(self) -> str:
        """返回当前分配的资源ID (用于日志)"""
        return self.worker_id 

    # =========================================================================
    # 3. 工具执行与同步桥接
    # =========================================================================

    def execute_tool(self, tool_name: str, params: Union[str, dict], **kwargs) -> Any:
        """
        重写执行入口。拦截所有工具调用，转发给 MCP Server。
        """
        # 参数标准化
        if isinstance(params, str):
            try: 
                params = json.loads(params)
            except: 
                params = {"arg": params}
        
        try:
            result = self._call_tool_sync(tool_name, params)
            
            # 保持返回值为 JSON 字符串格式 (兼容 Evaluation 逻辑)
            if isinstance(result, (dict, list)):
                return json.dumps(result, ensure_ascii=False)
            return str(result)
        except Exception as e:
            logger.error(f"Execute tool {tool_name} error: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    def _call_tool_sync(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """同步桥接器：运行异步 call_tool"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self._call_tool_async(tool_name, args))

    async def _call_tool_async(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """异步 MCP 调用实现"""
        # 注入 worker_id 以便 Server 路由
       
        tool_args = args.copy()
        
        # [修改] 强制覆盖 worker_id，不信任 LLM 传入的值
        # 无论 args 里有没有，都必须使用当前环境分配的 self.worker_id
        tool_args["worker_id"] = self.worker_id 

        sse_url = f"{self.server_url}/sse"
        
        try:
            async with sse_client(sse_url) as streams:
                async with ClientSession(streams[0], streams[1]) as session:
                    await session.initialize()
                    
                    result: CallToolResult = await session.call_tool(tool_name, tool_args)
                    
                    if result.content and len(result.content) > 0:
                        text_content = result.content[0]
                        if hasattr(text_content, "text"):
                            try:
                                return json.loads(text_content.text)
                            except json.JSONDecodeError:
                                return text_content.text
                        else:
                            # 处理图片等非文本内容
                            return str(text_content)
                    return {}
        except Exception as e:
            logger.error(f"MCP Async Call '{tool_name}' failed: {e}")
            raise e

    def _list_tools_sync(self):
        """同步桥接器：运行异步 list_tools"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self._list_tools_async())

    async def _list_tools_async(self):
        """异步获取工具列表"""
        sse_url = f"{self.server_url}/sse"
        async with sse_client(sse_url) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                result = await session.list_tools()
                return result.tools

    def env_close(self):
        """Worker 退出时的清理"""
        try:
            self._call_tool_sync("teardown_environment", {"worker_id": self.worker_id})
        except Exception:
            pass