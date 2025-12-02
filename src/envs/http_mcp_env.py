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
    "vm_computer_13": {
        "alloc": "setup_vm_session",
        "release": "teardown_environment",
        "alloc_args": ["config_name", "task_id"], # 除去 worker_id 外需要的参数
        "init_param_name": "init_script"  # VM 初始化脚本参数名
    },
    "vm_pyautogui": {
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
        
        # [新增] 初始化变量，用于保存初始观察数据
        self.initial_observation = None
        
        # [新增] 保存已分配的资源信息
        self.allocated_resources = {}
        
        # [新增] 标记是否已初始化工具
        self._tools_initialized = False

        # [新增] 初始化一个持久的事件循环
        import asyncio
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

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
        # [修改] 如果还没有建立连接，则不执行工具初始化
        if not self._tools_initialized:
            logger.info(f"[{self.worker_id}] Skipping tool initialization before connection is established")
            return
            
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
        # 标记工具可以初始化
        self._tools_initialized = True
        # 获取工具信息
        self._initialize_tools()

    # 需要添加到 HttpMCPEnv 类中
    def _run_sync(self, awaitable):
        """
        使用实例共享的事件循环运行异步任务，避免关闭导致 SSE 监听断开。
        """
        return self._loop.run_until_complete(awaitable)

    def _list_tools_sync(self):
        return self._run_sync(self.mcp_client.list_tools())

    def _call_tool_sync(self, name, arguments):
        # 自动注入 worker_id
        # 这是为了确保 Agent 调用 RAG 等工具时，后端能识别是哪个 Worker 发起的请求
        if isinstance(arguments, dict) and "worker_id" not in arguments:
            arguments["worker_id"] = self.worker_id
            
        # === [新增日志 START] ===
        logger.info(f"[{self.worker_id}] ⏳ Sync Calling: {name}...")
        start_time = time.time()
        # === [新增日志 END] ===

        res = self._run_sync(self.mcp_client.call_tool(name, arguments))

        # === [新增日志 START] ===
        duration = time.time() - start_time
        logger.info(f"[{self.worker_id}] ✅ Sync Call Done: {name} (Took {duration:.2f}s)")
        # === [新增日志 END] ===

        return res

    def _parse_mcp_response(self, response: CallToolResult) -> Dict[str, Any]:
        """解析MCP响应结果"""
        try:
            if response.content and len(response.content) > 0:
                # 检查内容类型并提取文本
                content_item = response.content[0]
                if hasattr(content_item, 'text'):
                    text_content = content_item.text
                elif hasattr(content_item, '__dict__') and 'text' in content_item.__dict__:
                    text_content = content_item.__dict__['text']
                else:
                    text_content = str(content_item)
                    
                if text_content:
                    return json.loads(text_content)
            return {"status": "unknown"}
        except Exception as e:
            logger.error(f"Failed to parse MCP response: {e}")
            return {"status": "error", "message": str(e)}

    def _log_alloc_failure(self, worker_id: str, res_type: str, data: Dict[str, Any], attempt: int):
        """记录资源分配失败日志"""
        error_msg = data.get("error", "Unknown error")
        logger.warning(f"Worker [{worker_id}] failed to allocate {res_type} (attempt {attempt+1}): {error_msg}")

    def _setup_single_resource(self, res_type: str, res_data: dict):
        self.allocated_resources[res_type] = res_data
        
        if res_type == "vm":
            self.vm_ip = res_data.get("ip")
            self.vm_port = res_data.get("port")
            
        elif res_type == "rag":
            self.rag_endpoint = res_data.get("endpoint")
            # ... 其他 RAG 初始化 ...

    # [新增] 获取初始观测的主入口
    def get_inital_obs(self) -> Dict[str, Any]:
        """
        调用系统工具 'get_batch_initial_observations' 获取所有资源的初始状态。
        """
        logger.info(f"[{self.worker_id}] Fetching batch initial observations from MCP...")
        
        combined_obs = {
            "vm": None,
            "rag": None,
            "raw_response": {}
        }

        try:
            # 1. 调用 MCP 工具
            res = self._call_tool_sync("get_batch_initial_observations", {"worker_id": self.worker_id})
            data = self._parse_mcp_response(res)
            combined_obs["raw_response"] = data

            if isinstance(data, dict) and "error" not in data:
                # 2. 兼容处理：更新 VM 数据到 self.initial_observation
                if "vm" in data and data["vm"]:
                    combined_obs["vm"] = data["vm"]
                    self.initial_observation = data["vm"]
                    logger.info(f"[{self.worker_id}] Initial VM observation updated.")
                
                # 3. 提取 RAG 数据
                if "rag" in data:
                    combined_obs["rag"] = data["rag"]
            else:
                logger.warning(f"[{self.worker_id}] Failed to fetch obs: {data.get('error')}")

            return combined_obs

        except Exception as e:
            logger.error(f"[{self.worker_id}] Exception in get_inital_obs: {e}")
            return combined_obs

    # [修改] allocate_resource：删除获取 obs 的旧逻辑
    def allocate_resource(self, worker_id: str, resource_init_data: Optional[Dict[str, Any]] = None) -> bool:
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
        
        # === [新增日志 START] ===
        logger.info(f"[{worker_id}] Resource Init Data Keys: {list(resource_init_data.keys()) if resource_init_data else 'None'}")
        # === [新增日志 END] ===
        
        # [修改] 清空上一轮的观察
        self.initial_observation = None
        
        # 如果有多种资源需要申请，尝试使用原子化申请
        if len(self.active_resources) > 1:
            return self._allocate_resources_atomically(resource_init_data)
        
        # 否则使用原有的逐个申请方式
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
                    data = self._parse_mcp_response(res)
                    
                    if data.get("status") != "success":
                        # 失败：记录日志，跳出内层循环，触发回滚
                        self._log_alloc_failure(worker_id, res_type, data, attempt)
                        all_success = False
                        break
                    
                    # [修改] 保存资源信息
                    self.allocated_resources[res_type] = data

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

    def _allocate_resources_atomically(self, resource_init_data: Dict[str, Any]) -> bool:
        """
        使用原子化方式申请资源（通过调用 Gateway 的系统工具）
        [优化] 不再直接访问 8000 端口，而是通过 8080 Gateway 的 'allocate_batch_resources' 工具转发
        """
        try:
            # 1. 构造工具参数
            # 注意：_call_tool_sync 会自动注入 'worker_id'，所以这里只需传业务参数
            # 但为了明确起见，显式传入也无妨
            args = {
                "resource_types": self.active_resources,
                "timeout": 600
            }
            
            logger.info(f"Worker [{self.worker_id}] calling MCP tool 'allocate_batch_resources' via Gateway...")

            # 2. 调用 MCP 工具 (走 SSE 通道 -> Gateway -> System Tools -> Resource API)
            # 这一步替代了原来的 requests.post
            res = self._call_tool_sync("allocate_batch_resources", args)
            
            # 3. 解析响应
            # Gateway 的工具返回的是 JSON 字符串，_parse_mcp_response 会将其转为字典
            data = self._parse_mcp_response(res)
            
            # 4. 错误处理
            # 检查是否是工具执行层面的错误 (例如 Resource API 返回 500 或超时)
            if isinstance(data, dict) and data.get("status") == "error":
                 logger.error(f"Atomic alloc tool failed: {data.get('message')}")
                 return False

            # 5. 处理成功的资源数据: {"vm": {...}, "rag": {...}}
            # 这一步逻辑与原来保持一致，因为 Gateway 透传了 Resource API 的返回结构
            for r_type, r_data in data.items():
                self._setup_single_resource(r_type, r_data)
            
            return True

        except Exception as e:
            logger.error(f"Failed to allocate resources atomically via MCP: {e}")
            return False

    def _setup_resources_logic(self, worker_id: str, init_data: Dict[str, Any]) -> bool:
        """
        设置资源初始化逻辑
        
        Args:
            worker_id: 工作进程ID
            init_data: 初始化数据
            
        Returns:
            bool: 设置是否成功
        """
        # 极简调用，不再解析 JSON
        try:
            res = self._call_tool_sync("setup_batch_resources", {
                "worker_id": worker_id, 
                "resource_init_configs": init_data
            })
            
            data = self._parse_mcp_response(res)
            return data.get("status") == "success"
        except Exception as e:
            logger.error(f"Failed to setup resources for worker {worker_id}: {e}")
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

    def env_close(self):
        # ... 其他清理逻辑 ...
        if hasattr(self, '_loop') and not self._loop.is_closed():
            self._loop.close()
        super().env_close()
