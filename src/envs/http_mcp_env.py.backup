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
    3. 支持多资源事务性分配 (VM + RAG)，实现全有或无(All-or-Nothing)。
    4. 自动从 Server 发现并注册工具。
    """
    
    # 开启重型资源模式，通知框架在 run_task 前后调用 allocate/release
    has_heavy_resource = True 

    def __init__(self, resource_manager=None, parallel_degree=1, **kwargs):
        # 1. 基础配置
        self.server_url = kwargs.get("mcp_server_url", "http://localhost:8080")
        self.resource_api_url = kwargs.get("resource_api_url", "http://localhost:8000")
        self.config_name = "default"
        
        # [新增] RAG 启用开关，默认为 False，可通过 env_kwargs 传入 True
        self.enable_rag = kwargs.get("enable_rag", False)
        
        # 2. 获取 worker_id (必需，作为会话标识)
        if "worker_id" in kwargs:
            self.worker_id = kwargs["worker_id"]
        else:
            import multiprocessing
            self.worker_id = multiprocessing.current_process().name
            
        logger.info(f"HttpMCPEnv initialized for {self.worker_id} -> {self.server_url} (SSE Mode). RAG Enabled: {self.enable_rag}")
        
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
            
            # 1. 过滤黑名单工具 (基础设施工具不暴露给 Agent)
            blacklist = {
                # VM 生命周期
                "setup_environment", 
                "teardown_environment", 
                # 辅助工具
                "get_observation", 
                "evaluate_task",
                # [新增] RAG 生命周期 (由环境自动管理，Agent 无需关心)
                "setup_rag_engine",
                "release_rag_engine"
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
        [核心修改] 事务性资源分配：
        1. 申请 VM
        2. 如果需要 RAG，申请 RAG
        3. 如果任一步骤失败，回滚已申请资源并重试
        """
        required_resources = ["VM"]
        if self.enable_rag:
            required_resources.append("RAG")
            
        logger.info(f"Worker [{worker_id}] requesting resources: {required_resources}...")
        
        retry_interval = 5
        max_retries = 100 
        
        for attempt in range(max_retries):
            vm_allocated = False
            rag_allocated = False
            
            try:
                # --- Step 1: 申请 VM (基础资源) ---
                res_vm = self._call_tool_sync("setup_environment", {
                    "config_name": self.config_name,
                    "task_id": "dynamic_alloc",
                    "worker_id": worker_id 
                })
                
                vm_status = self._parse_mcp_response(res_vm)
                if vm_status.get("status") != "success":
                    # VM 申请失败，记录日志并等待
                    self._log_alloc_failure(worker_id, "VM", vm_status, attempt)
                    time.sleep(retry_interval)
                    continue
                
                vm_allocated = True
                logger.info(f"Worker [{worker_id}] ✅ VM acquired.")

                # --- Step 2: 申请 RAG (可选资源) ---
                if self.enable_rag:
                    res_rag = self._call_tool_sync("setup_rag_engine", {
                        "worker_id": worker_id
                    })
                    
                    rag_status = self._parse_mcp_response(res_rag)
                    if rag_status.get("status") != "success":
                        # !!! RAG 申请失败，触发回滚 !!!
                        logger.warning(f"Worker [{worker_id}] ❌ RAG alloc failed ({rag_status.get('message')}). Rolling back VM...")
                        
                        # 回滚：释放刚刚申请到的 VM
                        self._call_tool_sync("teardown_environment", {"worker_id": worker_id})
                        vm_allocated = False
                        
                        # 等待重试
                        time.sleep(retry_interval)
                        continue
                        
                    rag_allocated = True
                    logger.info(f"Worker [{worker_id}] ✅ RAG acquired.")

                # --- 全部成功 ---
                return True

            except Exception as e:
                logger.error(f"Allocation critical exception: {e}")
                # 发生异常时的防御性清理
                if vm_allocated:
                    try: self._call_tool_sync("teardown_environment", {"worker_id": worker_id})
                    except: pass
                if rag_allocated:
                    try: self._call_tool_sync("release_rag_engine", {"worker_id": worker_id})
                    except: pass
                time.sleep(retry_interval)
        
        logger.error(f"Worker [{worker_id}] failed to acquire all resources after {max_retries} attempts.")
        return False

    def release_resource(self, worker_id: str, reset: bool = True) -> None:
        """
        任务结束后调用：释放所有资源
        """
        logger.info(f"Worker [{worker_id}] releasing resources...")
        
        # 1. 释放 RAG (如果启用了)
        if self.enable_rag:
            try:
                self._call_tool_sync("release_rag_engine", {"worker_id": worker_id})
            except Exception as e:
                logger.warning(f"RAG Release failed: {e}")

        # 2. 释放 VM
        try:
            self._call_tool_sync("teardown_environment", {"worker_id": worker_id})
        except Exception as e:
            logger.warning(f"VM Release failed: {e}")

    def get_allocated_resource_id(self) -> str:
        """返回当前分配的资源ID (用于日志)"""
        return self.worker_id 

    # =========================================================================
    # 3. 辅助方法
    # =========================================================================

    def _parse_mcp_response(self, res: Any) -> Dict[str, Any]:
        """解析 MCP 返回的 JSON 结果"""
        if isinstance(res, str):
            try: 
                return json.loads(res)
            except: 
                return {"status": "error", "message": res}
        return res if isinstance(res, dict) else {}

    def _log_alloc_failure(self, worker_id, res_type, status, attempt):
        """记录分配失败日志"""
        msg = status.get("message", "")
        # 如果是资源不足，且不是第一次尝试，降低日志级别以减少刷屏
        if ("No resources" in msg or "exhausted" in msg or "busy" in msg) and attempt % 5 != 0:
            return
        
        if "No resources" in msg or "exhausted" in msg or "busy" in msg:
             logger.info(f"Pool full ({res_type}), Worker [{worker_id}] waiting...")
        else:
             logger.warning(f"Worker [{worker_id}] {res_type} alloc error: {msg}")

    # =========================================================================
    # 4. 工具执行与同步桥接
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