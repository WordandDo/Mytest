# src/envs/http_mcp_env.py
import sys
import os
import json
import logging
import asyncio
import time
from typing import Dict, Any, Union, Optional, List, Tuple
from tools.tool import Tool

# 引入 MCP SDK
from mcp.types import CallToolResult

# 引入基类
from envs.enviroment import Environment

# 引入新的 MCP SSE 客户端
from utils.mcp_sse_client import MCPSSEClient

logger = logging.getLogger(__name__)

# --- 资源生命周期映射表 ---
# 定义了每种 resource_type 对应的申请(Alloc)和释放(Release)工具名称
# 这使得 Client 端可以根据配置动态调用正确的生命周期工具
RESOURCE_LIFECYCLE_MAP = {
    "vm": {
        "alloc": "setup_vm_session",
        "release": "teardown_environment",
        "alloc_args": ["config_name", "task_id"], # 除去 worker_id 外需要的参数
        "init_param_name": "init_script"  # VM 初始化脚本参数名
    },
    "rag": {
        "alloc": "setup_rag_session",
        "release": "release_rag_session",
        "alloc_args": [],
        "init_param_name": "rag_config"  # RAG 配置参数名
    }
}

class HttpMCPEnv(Environment):
    """
    配置驱动的 MCP 环境适配器
    
    特性：
    1. 读取 gateway_config.json 自动识别所需的资源模块。
    2. 动态、事务性地申请所有配置的资源。
    3. 自动根据配置过滤生命周期工具，将业务工具暴露给 Agent。
    """
    
    # 开启重型资源模式，通知框架在 run_task 前后调用 allocate/release
    has_heavy_resource = True 

    def __init__(self, resource_manager=None, parallel_degree=1, **kwargs):
        # 1. 基础配置
        self.server_url = kwargs.get("mcp_server_url", "http://localhost:8080")
        self.config_name = "default"
        
        # 2. 获取 worker_id
        if "worker_id" in kwargs:
            self.worker_id = kwargs["worker_id"]
        else:
            import multiprocessing
            self.worker_id = multiprocessing.current_process().name

        # 3. 实例化 MCP 客户端
        self.mcp_client = MCPSSEClient(f"{self.server_url}/sse")

        # 4. 加载 Gateway 配置 (模拟 src/mcp_server/main.py 的加载逻辑)
        # 优先使用 kwargs 中的配置路径，否则尝试默认路径
        config_path = kwargs.get("gateway_config_path", "gateway_config.json")
        self.modules_config = self._load_gateway_config(config_path)
        
        # 解析出需要管理的资源类型列表 (保持顺序)
        self.active_resources = [
            m.get("resource_type") 
            for m in self.modules_config.get("modules", [])
            if m.get("resource_type") in RESOURCE_LIFECYCLE_MAP
        ]

        logger.info(f"HttpMCPEnv initialized for {self.worker_id} -> {self.server_url}")
        logger.info(f"Active Resources from Config: {self.active_resources}")
        
        # 5. 调用父类初始化 (会触发 _initialize_tools)
        super().__init__(**kwargs)

    @property
    def mode(self) -> str:
        return "http_mcp"

    def _load_gateway_config(self, config_path: str) -> Dict[str, Any]:
        """加载服务器配置文件"""
        if not os.path.exists(config_path):
            logger.warning(f"Gateway config not found at {config_path}, using default VM-only config.")
            return {"modules": [{"resource_type": "vm"}]}
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load gateway config: {e}")
            return {"modules": [{"resource_type": "vm"}]}

    # =========================================================================
    # 1. 动态工具发现与初始化
    # =========================================================================

    def _initialize_tools(self):
        """
        根据配置动态生成工具 Schema 和描述。
        自动将生命周期管理工具加入黑名单，不暴露给 Agent。
        """
        try:
            logger.info(f"[{self.worker_id}] Fetching tools from MCP Server...")
            mcp_tools = self._list_tools_sync()
            
            # 1. 动态构建黑名单
            # 遍历所有启用的资源，将其 Alloc/Release 工具加入黑名单
            blacklist = set()
            blacklist.add("get_observation") # 辅助工具默认屏蔽
            blacklist.add("evaluate_task")
            
            for res_type in self.active_resources:
                lifecycle = RESOURCE_LIFECYCLE_MAP.get(res_type)
                if lifecycle:
                    blacklist.add(lifecycle["alloc"])
                    blacklist.add(lifecycle["release"])
            
            # 2. 过滤工具
            valid_tools = [t for t in mcp_tools if t.name not in blacklist]
            
            # 3. 生成 OpenAI Schema
            self.tool_schemas = [self._convert_mcp_tool_to_openai(t) for t in valid_tools]
            
            # 4. 生成工具描述 (System Prompt)
            descriptions = []
            for t in valid_tools:
                desc = t.description if t.description else "No description provided."
                descriptions.append(f"- {t.name}: {desc}")
            
            self.tool_descriptions = "\n".join(descriptions)
                
            logger.info(f"[{self.worker_id}] Initialized {len(valid_tools)} tools (Metadata only). Blacklisted: {len(blacklist)}")
            
        except Exception as e:
            logger.error(f"Failed to initialize tools: {e}")
            self.tool_schemas = []
            self.tool_descriptions = "Error loading tools."

    def _convert_mcp_tool_to_openai(self, mcp_tool) -> Dict[str, Any]:
        """将 MCP Tool 对象转换为 OpenAI Function Schema"""
        parameters = mcp_tool.inputSchema.copy() if hasattr(mcp_tool, "inputSchema") else {}
        
        # 移除 worker_id (由环境自动注入)
        if "properties" in parameters and "worker_id" in parameters["properties"]:
            del parameters["properties"]["worker_id"]
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
        # 禁止直接访问本地工具实例
        return None

    def list_tools(self) -> List[str]:
        return [s["function"]["name"] for s in self.tool_schemas]

    # =========================================================================
    # 2. 动态资源调度 (配置驱动)
    # =========================================================================

    def env_start(self):
        logger.info(f"Worker [{self.worker_id}] started (Config-Driven Mode)")
        # 建立长连接
        self._run_sync(self.mcp_client.connect())

    def allocate_resource(self, worker_id: str, resource_init_data: Dict[str, Any] = None) -> bool:
        """
        [核心重构] 动态事务性资源分配：
        遍历配置中的 active_resources，依次申请资源。
        如果任一步骤失败，回滚所有已申请资源。
        
        Args:
            resource_init_data: 过滤后的资源配置字典，例如:
                                {"vm": {"content": "..."}, "rag": {...}}
        """
        resource_init_data = resource_init_data or {}
        logger.info(f"Worker [{worker_id}] requesting resources: {self.active_resources}...")
        
        retry_interval = 5
        max_retries = 100 
        
        for attempt in range(max_retries):
            allocated_stack = [] # 记录已申请成功的资源，用于回滚
            all_success = True
            
            # --- 依次申请列表中的所有资源 ---
            for res_type in self.active_resources:
                lifecycle = RESOURCE_LIFECYCLE_MAP.get(res_type)
                if not lifecycle:
                    continue # 跳过未知资源
                
                tool_name = lifecycle["alloc"]
                
                # 构造参数
                args = {"worker_id": worker_id}
                if res_type == "vm":
                    args["config_name"] = self.config_name
                    args["task_id"] = "dynamic_alloc"
                
                # [关键]：注入 Task 特定的初始化数据
                if res_type in resource_init_data:
                    # 获取该资源对应的参数名 (需要在 RESOURCE_LIFECYCLE_MAP 中预定义)
                    # 例如 vm -> "init_script", rag -> "rag_config"
                    param_name = lifecycle.get("init_param_name") 
                    
                    if param_name:
                        # 提取 content，支持直接传字符串或 JSON dumps
                        config_content = resource_init_data[res_type].get("content", "")
                        if isinstance(config_content, (dict, list)):
                            import json
                            config_content = json.dumps(config_content)
                            
                        args[param_name] = config_content
                
                try:
                    # 调用 Alloc 工具
                    res = self._call_tool_sync(tool_name, args)
                    status = self._parse_mcp_response(res)
                    
                    if status.get("status") != "success":
                        # 失败：记录日志，跳出内层循环，触发回滚
                        self._log_alloc_failure(worker_id, res_type, status, attempt)
                        all_success = False
                        break
                    
                    # 成功：压入栈
                    allocated_stack.append(res_type)
                    if attempt == 0:
                        logger.info(f"Worker [{worker_id}] ✅ {res_type} acquired.")
                        
                except Exception as e:
                    logger.error(f"Alloc exception for {res_type}: {e}")
                    all_success = False
                    break
            
            # --- 结果判定 ---
            if all_success:
                return True
            else:
                # --- 回滚逻辑 (Rollback) ---
                if allocated_stack:
                    logger.warning(f"Worker [{worker_id}] Rolling back resources: {allocated_stack[::-1]}")
                    for res_type in reversed(allocated_stack):
                        lifecycle = RESOURCE_LIFECYCLE_MAP[res_type]
                        try:
                            self._call_tool_sync(lifecycle["release"], {"worker_id": worker_id})
                        except Exception as e:
                            logger.error(f"Rollback failed for {res_type}: {e}")
                
                # 等待重试
                time.sleep(retry_interval)
        
        logger.error(f"Worker [{worker_id}] failed to acquire all resources after {max_retries} attempts.")
        return False

    def release_resource(self, worker_id: str, reset: bool = True) -> None:
        """
        释放所有资源 (逆序释放)
        """
        logger.info(f"Worker [{worker_id}] releasing resources...")
        
        # 逆序遍历配置的资源，确保依赖关系正确的释放顺序 (例如先释放 RAG 再释放 VM)
        for res_type in reversed(self.active_resources):
            lifecycle = RESOURCE_LIFECYCLE_MAP.get(res_type)
            if lifecycle:
                try:
                    self._call_tool_sync(lifecycle["release"], {"worker_id": worker_id})
                except Exception as e:
                    logger.warning(f"{res_type} Release failed: {e}")

    def get_allocated_resource_id(self) -> str:
        return self.worker_id 

    # =========================================================================
    # 3. 辅助方法与底层通信 (重构)
    # =========================================================================

    def _parse_mcp_response(self, res: Any) -> Dict[str, Any]:
        if isinstance(res, str):
            try: return json.loads(res)
            except: return {"status": "error", "message": res}
        return res if isinstance(res, dict) else {}

    def _log_alloc_failure(self, worker_id, res_type, status, attempt):
        msg = status.get("message", "")
        # 减少刷屏
        if ("No resources" in msg or "exhausted" in msg or "busy" in msg) and attempt % 5 != 0:
            return
        if "No resources" in msg or "exhausted" in msg or "busy" in msg:
             logger.info(f"Pool full ({res_type}), Worker [{worker_id}] waiting...")
        else:
             logger.warning(f"Worker [{worker_id}] {res_type} alloc error: {msg}")

    def execute_tool(self, tool_name: str, params: Union[str, dict], **kwargs) -> Any:
        """
        [修改] 执行工具并应用多模态处理
        返回标准结构: {"text": str, "images": List[str]}
        """
        if isinstance(params, str):
            try: params = json.loads(params)
            except: params = {"arg": params}
        try:
            # 2. 调用 MCP 工具 (获取原始结果，通常是包含截图的 JSON 字符串)
            raw_result = self._call_tool_sync(tool_name, params)
            
            # 3. [新增] 调用多模态处理逻辑
            structured_result = self._process_multimodal_response(raw_result)
            return structured_result
            
        except Exception as e:
            logger.error(f"Execute tool {tool_name} error: {e}")
            # 出错时返回标准错误结构
            return {
                "text": json.dumps({"status": "error", "message": str(e)}),
                "images": []
            }

    def _process_multimodal_response(self, raw_response: Any) -> Dict[str, Any]:
        """
        [新增] 数据清洗与分离函数
        功能：
        1. 解析 JSON 字符串。
        2. 提取 'screenshot' 字段中的 Base64 图片。
        3. 将图片分离，避免其进入 Text Context。
        """
        # 默认返回值
        processed = {"text": str(raw_response), "images": []}
        
        # 尝试解析 JSON
        try:
            data = raw_response
            if isinstance(data, str):
                # 快速检查是否像 JSON，避免不必要的解析
                if not (data.strip().startswith("{") or data.strip().startswith("[")):
                    return processed
                data = json.loads(data)
            
            if not isinstance(data, dict):
                return processed

            # --- 提取图片逻辑 (适配 OSWorld 格式) ---
            # 目标结构通常是: {"status": "...", "observation": {"screenshot": "...", ...}}
            # 或者直接: {"screenshot": "...", ...}
            
            images = []
            
            # 递归或直接查找 screenshot 字段并移除它
            # 策略：为了保持原始数据的纯净，我们先复制一份
            # 注意：这里为了性能做了简化处理，主要针对 OSWorld 的结构
            
            # Case 1: 在 observation 内部
            if "observation" in data and isinstance(data["observation"], dict):
                obs = data["observation"]
                if "screenshot" in obs and obs["screenshot"]:
                    images.append(obs.pop("screenshot")) # 提取并从原字典中删除
            
            # Case 2: 在顶层
            elif "screenshot" in data and data["screenshot"]:
                images.append(data.pop("screenshot")) # 提取并从原字典中删除

            # 更新返回值
            # data 现在是移除了图片的"干净"字典，适合转为文本给 Tool Message
            processed["text"] = json.dumps(data, ensure_ascii=False)
            processed["images"] = images
            
            return processed

        except Exception as e:
            # 解析失败则回退到原始文本
            logger.warning(f"Failed to process multimodal response: {e}")
            return {"text": str(raw_response), "images": []}

    def _run_sync(self, coro):
        """统一的异步转同步辅助函数"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)

    def _call_tool_sync(self, tool_name: str, args: Dict[str, Any]) -> Any:
        return self._run_sync(self._call_tool_wrapper(tool_name, args))

    async def _call_tool_wrapper(self, tool_name: str, args: Dict[str, Any]) -> Any:
        tool_args = args.copy()
        tool_args["worker_id"] = self.worker_id
        return await self.mcp_client.call_tool(tool_name, tool_args)

    def _list_tools_sync(self):
        return self._run_sync(self._list_tools_wrapper())

    async def _list_tools_wrapper(self):
        return await self.mcp_client.list_tools()

    def env_close(self):
        self.release_resource(self.worker_id)