# src/envs/http_mcp_env.py
import sys
import os
import json
import logging
import asyncio
import time
from typing import Dict, Any, Union, Optional, List, Tuple
from datetime import datetime

# 保持原有引用
from tools.tool import Tool
from envs.data_models import Observation, TrajectoryStep, TaskTrajectory
from prompts.system_prompts import get_system_prompt as load_system_prompt

# 引入 MCP SDK
from mcp.types import CallToolResult
# 引入新的 MCP SSE 客户端
from utils.mcp_sse_client import MCPSSEClient

# [新增] 引入任务超时监控工具
from utils.task_timeout import TaskTimeoutMonitor, TaskTimeoutError, check_execution_timeout

import openai
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

logger = logging.getLogger(__name__)


class HttpMCPEnv:
    """
    配置驱动的 MCP 环境适配器 (独立解耦版)
    
    不再继承 Environment 基类，集成了所有必要的 Agent 执行与资源管理逻辑。
    """
    
    # 开启重型资源模式，通知框架在 run_task 前后调用 allocate/release
    has_heavy_resource = True 

    def __init__(self,
                 model_name: str = "gpt-4.1-2025-04-14",
                 parallel_degree=1,
                 **kwargs):
        
        # --- 原 Environment.__init__ 的逻辑 ---
        self.model_name = model_name
        self.config = kwargs
        
        # 工具管理
        self.tools: Dict[str, Tool] = {}
        self.tool_schemas: List[Dict[str, Any]] = []
        self.tool_descriptions: str = ""

        # --- 原 HttpMCPEnv.__init__ 的逻辑 ---
        
        # 1. 基础配置
        self.server_url = kwargs.get("mcp_server_url", "http://localhost:8000")
        self.config_name = "default"
        
        # 2. 获取 worker_id
        if "worker_id" in kwargs:
            self.worker_id = kwargs["worker_id"]
        else:
            import multiprocessing
            self.worker_id = multiprocessing.current_process().name

        # 3. 实例化 MCP 客户端
        self.mcp_client = MCPSSEClient(f"{self.server_url}/sse")

        # 4. 加载 Gateway 配置
        config_path = kwargs.get("gateway_config_path", "gateway_config.json")
        self.modules_config = self._load_gateway_config(config_path)

        # 解析出需要管理的资源类型列表
        # 直接从配置中读取启用的资源类型，无需映射表
        self.active_resources = [
            m.get("resource_type")
            for m in self.modules_config.get("modules", [])
            if m.get("resource_type")  # 只要有定义资源类型即可
        ]
        
        # 初始化状态变量
        self.initial_observation = None
        self.allocated_resources = {}
        self._tools_initialized = False

        # 初始化持久事件循环
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        logger.info(f"HttpMCPEnv initialized for {self.worker_id} -> {self.server_url}")
        
        # 触发工具初始化 (整合了原基类的调用)
        self._initialize_tools()

    @property
    def mode(self) -> str:
        return "http_mcp"

    # =========================================================================
    # 核心 Agent 执行逻辑 (从 Environment 迁移)
    # =========================================================================

    def run_task(self, task: Dict[str, Any], agent_config: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
        """
        执行完整的 Agent 任务循环
        [第2层超时] 在任务级别监控执行时间，超时自动抛出异常
        """
        task_id = task.get("id", "unknown")
        question = task.get("question", "")

        # 获取 Agent 配置参数
        model_name = agent_config.get("model_name", self.model_name)
        max_turns = agent_config.get("max_turns", 3)
        max_retries = agent_config.get("max_retries", 3)

        # [第2层超时] 获取任务超时配置，默认600秒
        task_timeout = float(
            agent_config.get("task_timeout",
            os.environ.get("TASK_EXECUTION_TIMEOUT", "600"))
        )

        # 获取任务输出目录（如果实现了该方法）
        task_output_dir = None
        if hasattr(self, "get_task_output_dir") and callable(self.get_task_output_dir):
            task_output_dir = self.get_task_output_dir(
                agent_config.get("output_dir", "results"),
                task_id,
                model_name
            )

        # [第2层超时] 使用超时监控器
        monitor = TaskTimeoutMonitor(task_timeout, task_id, self.worker_id)

        try:
            # 启动超时监控
            monitor.start()

            # 执行对话（将start_time和timeout传递给对话函数用于定期检查）
            messages = self._run_conversation(
                question, model_name, max_turns, max_retries, logger,
                task_timeout=task_timeout,
                task_start_time=time.time()
            )

            # 提取答案
            final_answer = self._extract_final_answer(messages)

            # 构建结果
            result = {
                "task_id": task_id,
                "question": question,
                "answer": final_answer,
                "messages": messages,
                "success": True,
                "error": None,
            }

            # 保存日志 (如果支持)
            if task_output_dir:
                self._save_conversation_log(
                    task_output_dir,
                    task_id,
                    question,
                    model_name,
                    messages,
                    result
                )

            return result

        except TaskTimeoutError as e:
            # 任务超时异常
            logger.error(f"❌ [TaskTimeout] Task {task_id} timeout: {e}")
            return {
                "task_id": task_id,
                "question": question,
                "answer": "",
                "messages": [],
                "success": False,
                "error": f"Task execution timeout: {e}",
            }

        except Exception as e:
            # 其他异常
            logger.error(f"❌ [TaskError] Task {task_id} failed: {e}")
            raise

        finally:
            # 取消超时监控
            monitor.cancel()

    def _run_conversation(self,
                         question: str,
                         model_name: str,
                         max_turns: int,
                         max_retries: int,
                         logger: logging.Logger,
                         task_timeout: Optional[float] = None,
                         task_start_time: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        执行 Agent 对话循环
        [第2层超时] 在每次循环前检查任务是否超时
        """
        system_prompt = self.get_system_prompt(question)
        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
        ]

        # 构建用户消息
        user_content: List[Dict[str, Any]] = [{"type": "text", "text": f"Question: {question}\n"}]

        # 注入初始观察
        initial_obs = getattr(self, "initial_observation", None)
        if initial_obs and isinstance(initial_obs, dict):
            # 添加截图
            screenshot_b64 = initial_obs.get("screenshot")
            if screenshot_b64:
                user_content.append({
                    "type": "text",
                    "text": "Here is the initial screen state of the computer:"
                })
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{screenshot_b64}",
                        "detail": "high"
                    }
                })

            # 添加 Accessibility Tree
            a11y_tree = initial_obs.get("accessibility_tree")
            if a11y_tree:
                user_content.append({
                    "type": "text",
                    "text": f"Accessibility Tree:\n{a11y_tree}"
                })

        messages.append({"role": "user", "content": user_content})

        client = self._get_openai_client()
        turn_count = 0

        while turn_count < max_turns:
            # [第2层超时] 在每轮开始前检查是否超时
            if task_timeout and task_start_time:
                if check_execution_timeout(task_start_time, task_timeout, "current_task", self.worker_id):
                    raise TaskTimeoutError(
                        f"Task timeout after {time.time() - task_start_time:.1f}s "
                        f"(limit: {task_timeout}s) at turn {turn_count}"
                    )

            retry = 0
            while retry < max_retries:
                try:
                    # 调用 LLM
                    print(f"Messages length: {len(messages)}")
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        tools=self.get_tool_schemas(),
                    )

                    if not hasattr(response, "choices") or not response.choices:
                        raise ValueError("OpenAI API returned empty response")

                    assistant_message = response.choices[0].message
                    # print(f"Assistant message: {assistant_message}")
                    messages.append(assistant_message.model_dump())

                    # 处理工具调用
                    if assistant_message.tool_calls:
                        # 修复 content 为 None 的情况
                        if messages[-1]['content'] is None:
                             messages[-1]['content'] = ""

                        for tool_call in assistant_message.tool_calls: # 处理所有 call 而不仅仅是 [:1]
                            tool_name = tool_call.function.name
                            tool_args = json.loads(tool_call.function.arguments)

                            print(f"Round {turn_count}: 🔧 Using tool: {tool_name}")
                            print(f"Round {turn_count}:    Arguments: {tool_args}")

                            # 【关键修改】直接调用 execute_tool，它现在适配了 MCP
                            tool_output = self.execute_tool(tool_name, tool_args)

                            # 解析标准化输出
                            if isinstance(tool_output, dict) and "images" in tool_output:
                                content_str = tool_output.get("text", "")
                                image_list = tool_output.get("images", [])
                            else:
                                content_str = str(tool_output)
                                image_list = []

                            print(f"Round {turn_count}:    Result: {content_str[:100]}... (Images: {len(image_list)})")

                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_name,
                                "content": content_str
                            })

                            # 注入 Vision 观察
                            if image_list:
                                user_content_blocks = []
                                user_content_blocks.append({
                                    "type": "text", 
                                    "text": f"Observation from tool '{tool_name}' (Screenshots):"
                                })
                                for img_b64 in image_list:
                                    user_content_blocks.append({
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{img_b64}",
                                            "detail": "high"
                                        }
                                    })
                                messages.append({
                                    "role": "user",
                                    "content": user_content_blocks
                                })
                        
                    else:
                        logger.info(f"Turn {turn_count}: final answer produced")
                        return messages 
                    
                    # 成功执行完一轮，跳出重试
                    break 

                except Exception as exc:
                    retry += 1
                    logger.warning(f"Retry {retry}/{max_retries} due to error: {exc}")
                    if retry >= max_retries:
                        raise
            turn_count += 1
            
        logger.warning("Max turns reached without final answer")
        return messages

    def _extract_final_answer(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """从对话历史中提取最终答案"""
        if not messages:
            return None
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    return content
                if content is not None:
                    return str(content)
        return None
    
    def _save_conversation_log(self, 
                             output_dir: str, 
                             task_id: str, 
                             question: str, 
                             model: str, 
                             messages: List[Dict[str, Any]], 
                             result: Dict[str, Any]):
        """
        保存详细的对话日志到 JSON 文件。
        
        结构包含：
        - 基础元数据 (task_id, model, timestamp)
        - 统计信息 (steps, status)
        - 完整结果 (result 对象)
        - 扁平化的对话历史 (便于人类阅读)
        """
        import os
        import json
        
        try:
            # 1. 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)
            
            # 2. 文件名安全处理 (防止 task_id 包含非法字符)
            safe_task_id = "".join([c if c.isalnum() or c in "-_." else "_" for c in str(task_id)])
            file_path = os.path.join(output_dir, f"{safe_task_id}.json")
            
            # 3. 计算一些基础统计信息
            assistant_turns = sum(1 for m in messages if m.get("role") == "assistant")
            
            # 4. 构造最终的日志对象
            # 我们将 result 包装在一个更有条理的结构中
            log_content = {
                "meta": {
                    "task_id": task_id,
                    "model_name": model,
                    "timestamp": datetime.now().isoformat(),
                    "output_file": file_path
                },
                "task": {
                    "question": question,
                    "status": "success" if result.get("success") else "failed",
                    "final_answer": result.get("answer"),
                    "total_turns": assistant_turns
                },
                # 保存原始的完整结果字典（包含 messages）
                "raw_result": result,
            }
            
            # 5. 写入文件
            with open(file_path, 'w', encoding='utf-8') as f:
                # default=str 用于处理某些可能无法序列化的对象（如自定义类实例）
                json.dump(log_content, f, ensure_ascii=False, indent=2, default=str)
                
            logger.info(f"[{self.worker_id}] ✅ Conversation log saved to: {file_path}")
            
        except Exception as e:
            logger.error(f"[{self.worker_id}] ❌ Failed to save conversation log: {e}")

    # =========================================================================
    # OpenAI Client 管理 (从 Environment 迁移)
    # =========================================================================

    def _get_openai_client(self) -> openai.OpenAI:
        if not hasattr(self, '_openai_client') or self._openai_client is None:
            api_key = self.config.get("openai_api_key") or os.environ.get("OPENAI_API_KEY", "")
            base_url = self.config.get("openai_api_url") or os.environ.get("OPENAI_API_URL") or os.environ.get("OPENAI_API_BASE")

            # [第1层超时] API调用超时配置
            # 单次API请求的超时时间，默认30秒
            timeout = float(self.config.get("openai_timeout", os.environ.get("OPENAI_TIMEOUT", "30")))
            # API请求失败后的最大重试次数，默认2次
            max_retries = int(self.config.get("openai_max_retries", os.environ.get("OPENAI_MAX_RETRIES", "2")))

            logger.info(f"[{self.worker_id}] Initializing OpenAI client with timeout={timeout}s, max_retries={max_retries}")

            openai.api_key = api_key
            if base_url:
                openai.base_url = base_url
                self._openai_client = openai.OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=timeout,
                    max_retries=max_retries
                )
            else:
                self._openai_client = openai.OpenAI(
                    api_key=api_key,
                    timeout=timeout,
                    max_retries=max_retries
                )
        return self._openai_client

    # =========================================================================
    # 工具管理与执行 (适配 MCP)
    # =========================================================================

    def execute_tool(self, tool_name: str, params: Union[str, dict], **kwargs) -> Union[str, Dict[str, Any]]:
        """
        [核心重构] 执行工具
        原 Environment 尝试本地调用 self.tools[name].call()。
        对于 HttpMCPEnv，所有工具都是远程的，因此直接代理到 _call_tool_sync。
        """
        # 兼容参数是字符串的情况 (有时候 LLM 会返回 JSON string)
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except:
                pass
        
        # 调用 MCP 同步接口
        return self._call_tool_sync(tool_name, params)

    def get_tool_schemas(self) -> List[ChatCompletionToolParam]:
        """获取用于 LLM API 的工具定义"""
        return self.tool_schemas  # type: ignore

    def get_tool_descriptions(self) -> str:
        """获取用于 Prompt 的工具描述文本"""
        return self.tool_descriptions

    def register_tool(self, tool: Tool):
        """为了保持接口兼容性保留，但 MCP 模式下通常不用本地注册"""
        self.tools[tool.name] = tool
        # 更新 metadata
        pass

    # =========================================================================
    # Prompt 工程 (从 Environment 迁移)
    # =========================================================================

    def get_action_space(self) -> Optional[str]:
        mode_config = self.config.get(self.mode)
        if isinstance(mode_config, dict) and "action_space" in mode_config:
            return mode_config.get("action_space")
        return self.config.get("action_space")

    def get_system_prompt(
        self,
        task_question: Optional[str] = None,
        extra_context: Optional[str] = None,
        action_space: Optional[str] = None,
    ) -> str:
        resolved_action_space = action_space or self.get_action_space()
        if resolved_action_space is None:
            prompt_template = load_system_prompt(environment_mode=self.mode)
        else:
            prompt_template = load_system_prompt(
                environment_mode=self.mode,
                action_space=resolved_action_space
            )

        prompt_with_tools = prompt_template.replace(
            "{tool_descriptions}",
            self.get_tool_descriptions()
        )

        prompt_with_placeholders = self._replace_prompt_placeholders(prompt_with_tools)

        suffix_parts: List[str] = []
        if task_question:
            suffix_parts.append(f"You are asked to complete the following task: {task_question}")
        if extra_context:
            suffix_parts.append(extra_context)

        if suffix_parts:
            prompt_with_placeholders = "\n".join([prompt_with_placeholders, *suffix_parts])

        return prompt_with_placeholders

    def _replace_prompt_placeholders(self, prompt: str) -> str:
        return prompt

    # =========================================================================
    # MCP 专用逻辑 (保持不变)
    # =========================================================================

    def _load_gateway_config(self, config_path: str) -> Dict[str, Any]:
        if not os.path.exists(config_path):
            logger.warning(f"Gateway config not found at {config_path}, using default VM-only config.")
            return {"modules": [{"resource_type": "vm"}]}
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load gateway config: {e}")
            return {"modules": [{"resource_type": "vm"}]}

    def _initialize_tools(self):
        """
        根据配置动态生成工具 Schema 和描述。
        """
        if not self._tools_initialized:
            logger.info(f"[{self.worker_id}] Skipping tool initialization before connection is established")
            return

        try:
            logger.info(f"[{self.worker_id}] Fetching tools from MCP Server...")
            mcp_tools = self._list_tools_sync()

            # 黑名单：Agent 不应直接调用的系统工具
            blacklist = {
                # 旧的评估和观察工具（已废弃或由系统自动调用）
                "get_observation",
                "evaluate_task",
                # 资源生命周期管理工具（由系统统一调用，Agent 不应直接使用）
                "allocate_batch_resources",
                "setup_batch_resources",
                "get_batch_initial_observations",
                "setup_vm_session",
                "setup_rag_session",
                "teardown_environment",
                "release_rag_session",
            }

            valid_tools = [t for t in mcp_tools if t.name not in blacklist]

            self.tool_schemas = [self._convert_mcp_tool_to_openai(t) for t in valid_tools]

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

    def _convert_mcp_tool_to_openai(self, mcp_tool) -> ChatCompletionToolParam:
        parameters = mcp_tool.inputSchema.copy() if hasattr(mcp_tool, "inputSchema") else {}
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

    def env_start(self):
        logger.info(f"Worker [{self.worker_id}] started (Config-Driven Mode)")
        self._run_sync(self.mcp_client.connect())
        self._tools_initialized = True
        self._initialize_tools()

    def env_close(self):
        if hasattr(self, '_loop') and not self._loop.is_closed():
            self._loop.close()

    def _run_sync(self, awaitable):
        return self._loop.run_until_complete(awaitable)

    def _list_tools_sync(self):
        return self._run_sync(self.mcp_client.list_tools())

    def _call_tool_sync(self, name, arguments) -> Union[Dict[str, Any], Any]:
        """同步调用 MCP 工具"""
        if isinstance(arguments, dict) and "worker_id" not in arguments:
            arguments["worker_id"] = self.worker_id
            
        logger.info(f"[{self.worker_id}] ⏳ Sync Calling: {name}...")
        start_time = time.time()

        res = self._run_sync(self.mcp_client.call_tool(name, arguments))

        duration = time.time() - start_time
        logger.info(f"[{self.worker_id}] ✅ Sync Call Done: {name} (Took {duration:.2f}s)")

        # 生命周期工具处理
        lifecycle_tools = {
            "allocate_batch_resources", "setup_batch_resources", 
            "get_batch_initial_observations", "setup_vm_session", 
            "setup_rag_session", "teardown_environment", "release_rag_session"
        }
        
        if name in lifecycle_tools:
            return res 

        # 通用输出标准化
        standardized_output = {
            "text": "",
            "images": []
        }
        text_parts = []
        
        if hasattr(res, 'content') and res.content:
            for item in res.content:
                if item.type == 'text':
                    text_parts.append(item.text)
                elif item.type == 'image':
                    standardized_output["images"].append(item.data)
                elif item.type == 'resource':
                    # 正确访问 EmbeddedResource 的 uri 属性
                    text_parts.append(f"[Resource: {item.resource.uri}]")
        else:
            text_parts.append(str(res) if res else "Success (No content)")

        standardized_output["text"] = "\n".join(text_parts)
        return standardized_output

    def _parse_mcp_response(self, response: CallToolResult) -> Dict[str, Any]:
        try:
            if response.content and len(response.content) > 0:
                content_item = response.content[0]
                # 正确访问 TextContent 的 text 属性
                if hasattr(content_item, 'text'):
                    text_content = content_item.text
                elif hasattr(content_item, 'resource') and hasattr(content_item.resource, 'text'):
                    text_content = content_item.resource.text
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


    def get_inital_obs(self) -> Dict[str, Any]:
        logger.info(f"[{self.worker_id}] Fetching batch initial observations from MCP...")
        combined_obs = {
            "vm": None,
            "rag": None,
            "raw_response": {}
        }
        try:
            res = self._call_tool_sync("get_batch_initial_observations", {"worker_id": self.worker_id})
            data = self._parse_mcp_response(res)
            combined_obs["raw_response"] = data

            if isinstance(data, dict) and "error" not in data:
                if "vm" in data and data["vm"]:
                    combined_obs["vm"] = data["vm"]
                    self.initial_observation = data["vm"]
                if "rag" in data:
                    combined_obs["rag"] = data["rag"]
            else:
                logger.warning(f"[{self.worker_id}] Failed to fetch obs: {data.get('error')}")
            return combined_obs
        except Exception as e:
            logger.error(f"[{self.worker_id}] Exception in get_inital_obs: {e}")
            return combined_obs

    def allocate_resource(self, worker_id: str, resource_init_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        统一的资源分配入口函数

        Args:
            worker_id: Worker ID
            resource_init_data: 资源初始化配置数据

        Returns:
            分配是否成功
        """
        resource_init_data = resource_init_data or {}
        logger.info(f"Worker [{worker_id}] requesting resources: {self.active_resources}...")

        self.initial_observation = None

        # 统一使用原子化批量分配
        return self._allocate_resources_atomically(resource_init_data)

    def _allocate_resources_atomically(self, resource_init_data: Dict[str, Any]) -> bool:
        """
        [修复版] 原子化资源申请 + 自动初始化
        """
        try:
            # 1. 申请资源 (Allocate)
            args = {
                "resource_types": self.active_resources,
                "timeout": 600
            }
            logger.info(f"Worker [{self.worker_id}] calling MCP tool 'allocate_batch_resources' via Gateway...")
            res = self._call_tool_sync("allocate_batch_resources", args)
            data = self._parse_mcp_response(res)

            if isinstance(data, dict) and data.get("status") == "error":
                 logger.error(f"Atomic alloc tool failed: {data.get('message')}")
                 return False

            # 2. 保存资源信息
            self.allocated_resources = data

            # 3. 执行资源初始化 (Setup)
            if resource_init_data:
                logger.info(f"[{self.worker_id}] Setting up resources...")
                setup_res = self._call_tool_sync("setup_batch_resources", {
                    "resource_init_configs": resource_init_data,
                    "allocated_resources": data
                })
                setup_result = self._parse_mcp_response(setup_res)

                if setup_result.get("status") not in ["success", "partial_error"]:
                    logger.error(f"[{self.worker_id}] Resource setup failed: {setup_result}")
                    self.release_resource(self.worker_id)
                    return False

            # 4. 获取初始观察
            self.get_inital_obs()
            return True

        except Exception as e:
            logger.error(f"Failed to allocate resources atomically via MCP: {e}")
            return False

    def release_resource(self, worker_id: str, reset: bool = True) -> None:
        """
        统一释放所有已分配的资源

        通过 Resource API 统一释放，MCP Server 端会自动调用相应的 teardown/release 工具

        Args:
            worker_id: Worker ID
            reset: 是否重置资源状态（保留参数以兼容旧接口）
        """
        logger.info(f"Worker [{worker_id}] releasing all resources...")

        # 遍历已分配的资源并逐一释放
        for res_type, res_data in list(self.allocated_resources.items()):
            resource_id = res_data.get("id")
            if resource_id:
                try:
                    # 直接调用 Resource API 释放，通过 MCP 同步调用
                    # 注意：这里需要使用同步的方式调用
                    import httpx
                    with httpx.Client() as client:
                        response = client.post(
                            f"{os.environ.get('RESOURCE_API_URL', 'http://localhost:8000')}/release",
                            json={"resource_id": resource_id, "worker_id": worker_id},
                            timeout=10.0
                        )
                        logger.info(f"Released {res_type} (resource_id={resource_id}): {response.status_code}")
                except Exception as e:
                    logger.warning(f"Failed to release {res_type} (resource_id={resource_id}): {e}")

        # 清空已分配资源记录
        self.allocated_resources.clear()
        logger.info(f"Worker [{worker_id}] resource cleanup completed")

    def get_allocated_resource_id(self) -> str:
        return self.worker_id