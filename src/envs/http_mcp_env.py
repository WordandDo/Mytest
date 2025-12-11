# src/envs/http_mcp_env.py
import sys
import os
import json
import logging
import asyncio
import time
import re
from typing import Dict, Any, Union, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass

# 引入 MCP SDK
from mcp.types import CallToolResult
# 引入 MCP SSE 客户端
from utils.mcp_sse_client import MCPSSEClient

# 引入任务超时监控工具
from utils.task_timeout import TaskTimeoutMonitor, TaskTimeoutError, check_execution_timeout

# 引入 system prompt 函数
from prompts.system_prompts import get_system_prompt

import openai
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

logger = logging.getLogger(__name__)

@dataclass
class ToolMetadata:
    """用于适配 GenericTrajectorySampler 的简单工具包装类"""
    name: str
    description: str
    parameters: List[Dict[str, Any]]

class HttpMCPEnv:
    """
    配置驱动的 MCP 环境适配器 (MCP 纯净版)
    
    完全基于 Model Context Protocol (MCP) 与远程 Gateway/Server 交互。
    负责 Agent 执行循环、工具调用转发以及通过 MCP 工具进行资源生命周期管理。
    """
    
    # 开启重型资源模式，通知框架在 run_task 前后调用 allocate/release
    has_heavy_resource = True 

    def __init__(self,
                 model_name: str = "gpt-4.1-2025-04-14",
                 parallel_degree=1,
                 **kwargs):
        
        self.model_name = model_name
        self.config = kwargs
        
        # 工具元数据缓存
        self.tool_schemas: List[Dict[str, Any]] = []
        self.tool_descriptions: str = ""
        # [新增] 本地工具缓存，用于支持 get_tool
        self.local_tools: Dict[str, ToolMetadata] = {}

        # 1. 基础配置
        self.server_url = kwargs.get("mcp_server_url", "http://localhost:8080")
        
        # 2. 获取 worker_id
        if "worker_id" in kwargs:
            self.worker_id = kwargs["worker_id"]
        else:
            import multiprocessing
            self.worker_id = multiprocessing.current_process().name

        # 3. 实例化 MCP 客户端
        self.mcp_client = MCPSSEClient(f"{self.server_url}/sse")

        # 4. 加载 Gateway 配置 (确定需要申请哪些资源)
        config_path = kwargs.get("gateway_config_path", "gateway_config.json")
        self.modules_config = self._load_gateway_config(config_path)

        # [修复] 解析活动资源类型时，过滤掉不需要后端分配的 'system' 类型
        # 'system' 通常指代无状态的系统工具集，不需要向 Resource API 申请锁定
        self.active_resources = [
            m.get("resource_type")
            for m in self.modules_config.get("modules", [])
            if m.get("resource_type") and m.get("resource_type") != "system"
        ]
        
        # 初始化状态变量
        self.initial_observation = None
        self.allocated_resources = {}
        self._tools_initialized = False
        # 顺序编号：用于为工具产生的图片生成连续的 <obs_i> token
        self._obs_counter = 0

        # 5. 工具白名单（优先策略）
        # 优先使用传入的 tool_whitelist；否则尝试从环境变量 MCP_TOOL_WHITELIST 读取（逗号分隔）
        whitelist_arg = kwargs.get("tool_whitelist")
        if isinstance(whitelist_arg, (list, tuple, set)):
            self._tool_whitelist = {str(x).strip() for x in whitelist_arg if str(x).strip()}
        elif isinstance(whitelist_arg, str):
            self._tool_whitelist = {x.strip() for x in whitelist_arg.split(',') if x.strip()}
        else:
            env_wl = os.environ.get("MCP_TOOL_WHITELIST", "")
            self._tool_whitelist = {x.strip() for x in env_wl.split(',') if x.strip()} if env_wl else set()

        # 初始化持久事件循环
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        logger.info(f"HttpMCPEnv initialized: {self.worker_id} -> {self.server_url}, resources: {self.active_resources}")
        
        # 初始化远程工具列表
        self._initialize_tools()

    @property
    def mode(self) -> str:
        return "http_mcp"

    # =========================================================================
    # [新增/修复] 核心适配接口：get_tool 和 list_tools
    # =========================================================================

    def list_tools(self) -> List[str]:
        """返回所有可用工具的名称列表"""
        return list(self.local_tools.keys())

    def get_tool(self, tool_name: str) -> Optional[ToolMetadata]:
        """
        获取工具对象（适配器模式）。
        GenericTrajectorySampler 需要访问 tool.name, tool.description, tool.parameters
        """
        return self.local_tools.get(tool_name)

    # =========================================================================
    # 核心 Agent 执行逻辑
    # =========================================================================

    def run_task(self, task: Dict[str, Any], agent_config: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
        """
        执行完整的 Agent 任务循环
        """
        task_id = task.get("id", "unknown")
        question = task.get("question", "")

        model_name = agent_config.get("model_name", self.model_name)
        max_turns = agent_config.get("max_turns", 3)
        max_retries = agent_config.get("max_retries", 3)

        # 获取任务超时配置
        task_timeout = float(
            agent_config.get("task_timeout",
            os.environ.get("TASK_EXECUTION_TIMEOUT", "600"))
        )

        task_output_dir = None
        if hasattr(self, "get_task_output_dir") and callable(self.get_task_output_dir):
            task_output_dir = self.get_task_output_dir(
                agent_config.get("output_dir", "results"),
                task_id,
                model_name
            )

        monitor = TaskTimeoutMonitor(task_timeout, task_id, self.worker_id)

        try:
            monitor.start()

            messages = self._run_conversation(
                question, model_name, max_turns, max_retries, logger,
                task_timeout=task_timeout,
                task_start_time=time.time()
            )

            final_answer = self._extract_final_answer(messages)

            result = {
                "task_id": task_id,
                "question": question,
                "answer": final_answer,
                "messages": messages,
                "success": True,
                "error": None,
            }

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
            logger.error(f"❌ [TaskError] Task {task_id} failed: {e}")
            raise

        finally:
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
        
        system_prompt = self.get_system_prompt(question)
        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
        ]

        user_content: List[Dict[str, Any]] = [{"type": "text", "text": f"Question: {question}\n"}]

        # 注入初始观察
        initial_obs = getattr(self, "initial_observation", None)
        
        # === 减少日志：移除初始观察状态检查 ===
        # if initial_obs:
        #     logger.info(f"[{self.worker_id}] [LLM_INJECT_LOG] Initial Observation Status: Present for injection.")
        # else:
        #     logger.info(f"[{self.worker_id}] [LLM_INJECT_LOG] Initial Observation Status: Not present for injection.")

        if initial_obs and isinstance(initial_obs, dict):
            if initial_obs.get("screenshot"):
                user_content.append({
                    "type": "text",
                    "text": "Here is the initial screen state of the computer:"
                })
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{initial_obs['screenshot']}",
                        "detail": "high"
                    }
                })

            if initial_obs.get("accessibility_tree"):
                user_content.append({
                    "type": "text",
                    "text": f"Accessibility Tree:\n{initial_obs['accessibility_tree']}"
                })

        # [新增] 注入任务输入图片（如由子类设置的 self.input_images）
        # 支持两种形式：
        # - base64 数据：{"b64": "..."}
        # - 远程 URL：{"url": "https://..."}
        # 若同时存在，则都注入，便于模型预览与工具调用（例如反向图搜需要 URL）
        # 注意：图片标记使用成对标签，例如 <image_1> ... </image_1>
        input_images = getattr(self, "input_images", None)
        if isinstance(input_images, list) and input_images:
            for idx, img in enumerate(input_images, start=1):
                open_token = f"<image_{idx}>"
                close_token = f"</image_{idx}>"
                if isinstance(img, dict):
                    b64 = img.get("b64")
                    url = img.get("url")
                    # 始终先输出纯 Token 行，确保匹配提取逻辑 (<token> 必须是文本项最后)
                    user_content.append({"type": "text", "text": open_token})
                    if b64:
                        user_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64}",
                                "detail": "high"
                            }
                        })
                    elif url:
                        user_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": url,
                                "detail": "high"
                            }
                        })
                    # 结尾关闭 Token（供可读性；裁切索引不依赖该行）
                    user_content.append({"type": "text", "text": close_token})

        messages.append({"role": "user", "content": user_content})

        # === 减少日志：移除用户消息内容检查 ===
        # safe_msg = self._truncate_data(messages[1], max_len=200)
        # logger.info(f"[{self.worker_id}] [LLM_INJECT_LOG] First User Message Content (Check Injection): {json.dumps(safe_msg, indent=2, ensure_ascii=False)}")

        client = self._get_openai_client()
        turn_count = 0

        while turn_count < max_turns:
            if task_timeout and task_start_time:
                if check_execution_timeout(task_start_time, task_timeout, "current_task", self.worker_id):
                    raise TaskTimeoutError(
                        f"Task timeout after {time.time() - task_start_time:.1f}s "
                        f"(limit: {task_timeout}s) at turn {turn_count}"
                    )

            retry = 0
            while retry < max_retries:
                try:
                    # 减少日志：仅在需要时输出
                    # logger.info(f"Turn {turn_count}: Calling LLM...")
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        tools=self.get_tool_schemas(),
                    )

                    if not hasattr(response, "choices") or not response.choices:
                        raise ValueError("OpenAI API returned empty response")

                    assistant_message = response.choices[0].message
                    messages.append(assistant_message.model_dump())

                    if assistant_message.tool_calls:
                        if messages[-1]['content'] is None:
                             messages[-1]['content'] = ""

                        for tool_call in assistant_message.tool_calls:
                            tool_name = tool_call.function.name
                            tool_args = json.loads(tool_call.function.arguments)

                            # 减少日志：仅输出工具名称
                            logger.info(f"🔧 {tool_name}")

                            # 代理到 MCP 执行
                            tool_output = self.execute_tool(tool_name, tool_args)

                            if isinstance(tool_output, dict) and "images" in tool_output:
                                content_str = tool_output.get("text", "")
                                image_list = tool_output.get("images", [])
                            else:
                                content_str = str(tool_output)
                                image_list = []

                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_name,
                                "content": content_str
                            })

                            if image_list:
                                # 将工具返回的图片统一包裹为 <obs_i> ... </obs_i>
                                obs_blocks = []
                                for img_b64 in image_list:
                                    self._obs_counter += 1
                                    open_obs = f"<obs_{self._obs_counter}>"
                                    close_obs = f"</obs_{self._obs_counter}>"
                                    # 开标签
                                    obs_blocks.append({"type": "text", "text": open_obs})
                                    # 图片
                                    obs_blocks.append({
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{img_b64}",
                                            "detail": "high"
                                        }
                                    })
                                    # 闭标签（可读性）
                                    obs_blocks.append({"type": "text", "text": close_obs})
                                messages.append({"role": "user", "content": obs_blocks})

                    else:
                        # 减少日志：最终答案产生时不再输出
                        # logger.info(f"Turn {turn_count}: final answer produced")
                        return messages
                    
                    break # 成功则跳出重试循环

                except Exception as exc:
                    retry += 1
                    logger.warning(f"Retry {retry}/{max_retries} due to error: {exc}")
                    if retry >= max_retries:
                        raise
            turn_count += 1
            
        logger.warning("Max turns reached without final answer")
        return messages

    def _extract_final_answer(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """
        Extract final answer from messages.
        First tries to extract content within <FINAL_ANSWER> tags.
        Falls back to returning the last assistant message if tags not found.
        """
        if not messages:
            return None

        # Search for messages with FINAL_ANSWER tags (from newest to oldest)
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content")
                if isinstance(content, str):
                    # Try to extract answer from special tokens
                    match = re.search(r'<FINAL_ANSWER>(.*?)</FINAL_ANSWER>', content, re.DOTALL)
                    if match:
                        answer = match.group(1).strip()
                        if answer:
                            return answer

        # Fallback: return last assistant message if no tags found
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    return content
                if content is not None:
                    return str(content)

        return None
    
    def _save_conversation_log(self, output_dir, task_id, question, model, messages, result):
        import os
        import json
        try:
            os.makedirs(output_dir, exist_ok=True)
            safe_task_id = "".join([c if c.isalnum() or c in "-_." else "_" for c in str(task_id)])
            file_path = os.path.join(output_dir, f"{safe_task_id}.json")
            assistant_turns = sum(1 for m in messages if m.get("role") == "assistant")
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
                "raw_result": result,
            }
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(log_content, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"[{self.worker_id}] ✅ Conversation log saved to: {file_path}")
        except Exception as e:
            logger.error(f"[{self.worker_id}] ❌ Failed to save conversation log: {e}")

    # =========================================================================
    # OpenAI Client 管理
    # =========================================================================

    def _get_openai_client(self) -> openai.OpenAI:
        if not hasattr(self, '_openai_client') or self._openai_client is None:
            api_key = self.config.get("openai_api_key") or os.environ.get("OPENAI_API_KEY", "")
            base_url = self.config.get("openai_api_url") or os.environ.get("OPENAI_API_URL") or os.environ.get("OPENAI_API_BASE")
            timeout = float(self.config.get("openai_timeout", os.environ.get("OPENAI_TIMEOUT", "30")))
            max_retries = int(self.config.get("openai_max_retries", os.environ.get("OPENAI_MAX_RETRIES", "2")))

            # 减少日志：移除OpenAI client初始化日志
            # logger.info(f"[{self.worker_id}] Initializing OpenAI client...")

            openai.api_key = api_key
            if base_url:
                openai.base_url = base_url
                self._openai_client = openai.OpenAI(
                    api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries
                )
            else:
                self._openai_client = openai.OpenAI(
                    api_key=api_key, timeout=timeout, max_retries=max_retries
                )
        return self._openai_client

    # =========================================================================
    # 工具管理与执行 (适配 MCP)
    # =========================================================================

    def execute_tool(self, tool_name: str, params: Union[str, dict], **kwargs) -> Union[str, Dict[str, Any]]:
        """执行工具：直接代理到 MCP"""
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except:
                pass
        return self._call_tool_sync(tool_name, params)

    def get_tool_schemas(self) -> List[ChatCompletionToolParam]:
        return self.tool_schemas  # type: ignore

    def get_tool_descriptions(self) -> str:
        return self.tool_descriptions

    # =========================================================================
    # Prompt 工程
    # =========================================================================

    def get_action_space(self) -> Optional[str]:
        mode_config = self.config.get(self.mode)
        if isinstance(mode_config, dict) and "action_space" in mode_config:
            return mode_config.get("action_space")
        return self.config.get("action_space")

    def get_system_prompt(self, task_question: Optional[str] = None, **kwargs) -> str:
        action_space = self.get_action_space()
        if action_space is None:
            prompt_template = get_system_prompt(environment_mode=self.mode)
        else:
            prompt_template = get_system_prompt(
                environment_mode=self.mode, action_space=action_space
            )

        prompt = prompt_template.replace("{tool_descriptions}", self.get_tool_descriptions())

        if task_question:
            prompt += f"\nYou are asked to complete the following task: {task_question}"
        
        return prompt

    # =========================================================================
    # MCP 专用逻辑
    # =========================================================================

    def _load_gateway_config(self, config_path: str) -> Dict[str, Any]:
        if not os.path.exists(config_path):
            return {"modules": []}
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {"modules": []}

    def _initialize_tools(self):
        """从 MCP Server 获取工具列表并进行本地适配"""
        if not self._tools_initialized:
            return

        try:
            # 减少日志：移除工具获取日志
            # logger.info(f"[{self.worker_id}] Fetching tools from MCP Server...")
            mcp_tools = self._list_tools_sync()

            # 1. 默认为白名单优先（如未配置白名单，则回退到黑名单与隐藏标记过滤）
            default_blacklist = {
                "get_observation", "evaluate_task",
                "allocate_batch_resources", "setup_batch_resources",
                "get_batch_initial_observations", "setup_vm_session",
                "setup_rag_session", "teardown_rag_session", "teardown_environment", "release_rag_session",
            }

            valid_tools = []
            self.local_tools = {}

            for t in mcp_tools:
                # 1) 白名单优先：若配置了白名单，则仅允许白名单内的工具
                if self._tool_whitelist:
                    if t.name not in self._tool_whitelist:
                        continue
                else:
                    # 2) 未设置白名单时，使用黑名单 + [HIDDEN] 过滤作为兜底策略
                    if t.name in default_blacklist:
                        continue
                    description = t.description or ""
                    if description.startswith("[HIDDEN]"):
                        continue
                
                valid_tools.append(t)
                
                # [核心适配] 将 MCP Schema 转换为 ToolMetadata (List[Dict] 格式)
                # 这允许 GenericTrajectorySampler 能够读取 tool.parameters
                converted_params = self._convert_schema_to_params(t.inputSchema)
                self.local_tools[t.name] = ToolMetadata(
                    name=t.name,
                    description=t.description or "",
                    parameters=converted_params
                )

            # 生成 Schema 和描述字符串给 LLM
            self.tool_schemas = [self._convert_mcp_tool_to_openai(t) for t in valid_tools]

            descriptions = [f"- {t.name}: {t.description or 'No description.'}" for t in valid_tools]
            self.tool_descriptions = "\n".join(descriptions)

            # 减少日志：移除工具数量日志
            # logger.info(f"[{self.worker_id}] {len(valid_tools)} tools initialized")

        except Exception as e:
            logger.error(f"[{self.worker_id}] Failed to initialize tools: {e}")

    def _convert_schema_to_params(self, schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        """将 JSON Schema properties 转换为参数列表格式"""
        params = []
        if not schema or "properties" not in schema:
            return params
            
        required_set = set(schema.get("required", []))
        properties = schema.get("properties", {})
        
        for name, prop in properties.items():
            if name == "worker_id": 
                continue
            
            param_def = {
                "name": name,
                "type": prop.get("type", "string"),
                "description": prop.get("description", ""),
                "required": name in required_set
            }
            if param_def["type"] == "array" and "items" in prop:
                 param_def["array_type"] = prop["items"].get("type", "string")
                 
            params.append(param_def)
        return params

    def _convert_mcp_tool_to_openai(self, mcp_tool) -> ChatCompletionToolParam:
        """
        将 MCP 工具定义转换为 OpenAI 工具格式。
        自动移除 worker_id 参数，因为它由环境自动注入。
        """
        # 深拷贝避免修改原始 schema
        parameters = {}
        if hasattr(mcp_tool, "inputSchema") and mcp_tool.inputSchema:
            parameters = mcp_tool.inputSchema.copy()
        
        # 移除 worker_id 参数（环境自动注入）
        if "properties" in parameters and "worker_id" in parameters["properties"]:
            del parameters["properties"]["worker_id"]
        if "required" in parameters and "worker_id" in parameters["required"]:
            parameters["required"] = [p for p in parameters["required"] if p != "worker_id"]

        return {
            "type": "function",
            "function": {
                "name": mcp_tool.name,
                "description": mcp_tool.description or "No description provided.",
                "parameters": parameters
            }
        }

    def env_start(self):
        logger.info(f"Worker [{self.worker_id}] connecting to MCP...")
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

    def _call_tool_sync(self, name: str, arguments: Union[Dict[str, Any], str]):
        """
        同步调用 MCP 工具
        
        自动注入 worker_id 参数（如果缺失）以确保工具调用的一致性。
        特殊处理资源管理类工具的返回值。
        """
        # 确保参数是字典格式
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON arguments for tool {name}: {arguments}")
                raise ValueError(f"Invalid JSON arguments for tool {name}")

        # 自动注入 worker_id（如果缺失）
        if isinstance(arguments, dict) and "worker_id" not in arguments:
            arguments["worker_id"] = self.worker_id

        # 记录调用日志（截断长参数）
        try:
            if isinstance(arguments, dict):
                safe_args = dict(arguments)
                if "messages" in safe_args:
                    msgs = safe_args["messages"]
                    safe_args["messages"] = f"[len={len(msgs)}]"
            else:
                safe_args = arguments
            logger.info(f"[{self.worker_id}] 🔧 Tool call -> {name} args={safe_args}")
        except Exception:
            pass

        # 发起同步工具调用
        try:
            res: CallToolResult = self._run_sync(self.mcp_client.call_tool(name, arguments))
        except Exception as e:
            logger.error(f"[{self.worker_id}] ❌ Tool call failed -> {name}: {e}")
            # 返回标准化错误结构，避免上层崩溃
            return {"text": json.dumps({"status": "error", "tool": name, "message": str(e)}, ensure_ascii=False), "images": []}

        # 特殊处理资源管理类工具（直接返回原始结果）
        resource_management_tools = {
            "allocate_batch_resources", "setup_batch_resources",
            "get_batch_initial_observations", "teardown_environment",
            "release_batch_resources"
        }
        if name in resource_management_tools:
            return res

        # 标准化输出格式（文本+图像）
        output = {
            "text": "",
            "images": []
        }

        # 解析 MCP 响应内容
        texts = []
        if hasattr(res, 'content') and res.content:
            for item in res.content:
                # 文本内容累积
                if item.type == 'text':
                    texts.append(item.text)
                # 图像内容收集
                elif item.type == 'image':
                    # 支持 Data URI 和纯 Base64 两种格式
                    image_data = item.data
                    if ',' in image_data:
                        # Data URI 格式: data:image/png;base64,...
                        image_data = image_data.split(',', 1)[1]
                    output["images"].append(image_data)
        else:
            # 无内容时返回默认成功消息
            texts.append(str(res) if res else "Success")

        # 合并所有文本内容
        output_text = "\n".join(texts)
        output["text"] = output_text

        # 记录返回摘要（截断）
        try:
            preview = output_text[:200].replace("\n", " ") if output_text else ""
            logger.info(f"[{self.worker_id}] ✅ Tool result <- {name} text='{preview}' images={len(output['images'])}")
            # 检测结构化错误并打印
            try:
                data = json.loads(output_text)
                if isinstance(data, dict) and data.get("status") == "error":
                    logger.error(f"[{self.worker_id}] ❗ Tool error <- {name}: {data.get('message')}")
            except Exception:
                pass
        except Exception:
            pass
        return output

    def _parse_mcp_response(self, response: CallToolResult) -> Dict[str, Any]:
        try:
            if response.content and len(response.content) > 0:
                content_item = response.content[0]
                text_content = getattr(content_item, 'text', None)
                if not text_content and hasattr(content_item, 'resource'):
                    text_content = getattr(content_item.resource, 'text', None)
                
                if text_content:
                    try:
                        data = json.loads(text_content)
                        if isinstance(data, dict) and data.get("status") == "error":
                            logger.error(f"[{self.worker_id}] Tool returned error payload: {data}")
                        return data
                    except Exception as e:
                        logger.error(f"[{self.worker_id}] Failed to parse MCP response JSON: {e}")
                        return {"status": "error", "message": str(e), "raw": text_content}
            return {"status": "unknown"}
        except Exception as e:
            logger.error(f"[{self.worker_id}] Exception parsing MCP response: {e}")
            return {"status": "error", "message": str(e)}

    def get_inital_obs(self) -> Dict[str, Any]:
        """调用 MCP 获取初始观察，并应用黑名单过滤"""
        # 减少日志：移除初始观察获取日志
        # logger.info(f"[{self.worker_id}] Fetching initial observations...")

        combined_obs = {}
        self.initial_observation = None # 重置主观察

        # 1. 从 self.config (已合并任务 metadata) 获取黑名单设置
        # resource_blacklist: 资源类型黑名单列表，例如 ['rag', 'vm_pyautogui']
        resource_blacklist = self.config.get("observation_blacklist", [])
        # content_blacklist: 资源内容细粒度黑名单，例如 {'vm_computer_13': ['accessibility_tree']}
        content_blacklist = self.config.get("observation_content_blacklist", {})

        try:
            # 调用系统工具获取所有资源的初始观察
            res = self._call_tool_sync("get_batch_initial_observations", {"worker_id": self.worker_id})
            data = self._parse_mcp_response(res)

            # === 减少日志：移除原始观察数据和过滤后观察数据的详细日志 ===
            # safe_data = self._truncate_data(data, max_len=100)
            # logger.info(f"[{self.worker_id}] [OBS_LOG] Raw observation data from MCP (Truncated): {json.dumps(safe_data, indent=2, ensure_ascii=False)}")

            if isinstance(data, dict) and "error" not in data:
                # 2. 遍历并应用黑名单过滤
                for resource_type, obs_content in data.items():
                    # A. 资源类型黑名单过滤
                    if resource_type in resource_blacklist:
                        # 减少日志：移除黑名单跳过日志
                        # logger.info(f"[{self.worker_id}] Blacklisted resource observation skipped: {resource_type}")
                        continue
                        
                    # B. 观察内容细粒度过滤
                    filtered_obs_content = obs_content
                    if resource_type in content_blacklist and isinstance(obs_content, dict):
                        # 对资源内容进行拷贝，避免修改原始数据
                        filtered_obs_content = obs_content.copy()
                        keys_to_remove = content_blacklist[resource_type]
                        
                        for key in keys_to_remove:
                            if key in filtered_obs_content:
                                del filtered_obs_content[key]
                                # 减少日志：移除黑名单内容移除日志
                                # logger.info(f"[{self.worker_id}] Blacklisted observation content removed: {resource_type}.{key}")

                    combined_obs[resource_type] = filtered_obs_content

                    # 3. 动态确定主要观察 (用于 LLM 注入)
                    # 优先将视觉环境（vm/desktop相关的）且非空的观察设为主观察
                    # 注意：这里使用启发式检查，同时过滤掉内容为空或只剩少量键的观察
                    if self.initial_observation is None and ("vm" in resource_type.lower() or "desktop" in resource_type.lower()):
                         # 确保过滤后仍包含用于主观察的必要内容 (如 screenshot)
                         if filtered_obs_content and any(key in filtered_obs_content for key in ["screenshot", "accessibility_tree", "text"]):
                            self.initial_observation = filtered_obs_content
                         
            else:
                logger.warning(f"[{self.worker_id}] Failed to fetch obs: {data.get('error')}")

            # === 减少日志：移除最终观察和主观察的详细日志 ===
            # safe_obs = self._truncate_data(combined_obs, max_len=100)
            # logger.info(f"[{self.worker_id}] [OBS_LOG] Final combined observations (Filtered & Truncated): {json.dumps(safe_obs, indent=2, ensure_ascii=False)}")
            # if self.initial_observation:
            #     logger.info(f"[{self.worker_id}] [OBS_LOG] Primary initial_observation SET. Keys: {list(self.initial_observation.keys())}")
            # else:
            #     logger.info(f"[{self.worker_id}] [OBS_LOG] Primary initial_observation is None.")

            return combined_obs
        except Exception as e:
            logger.error(f"[{self.worker_id}] Obs fetch error: {e}")
            # 即使失败，也要返回已收集的部分观察结果
            return combined_obs

    def allocate_resource(self, worker_id: str, resource_init_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        统一的资源分配入口函数 (MCP 模式)
        """
        resource_init_data = resource_init_data or {}
        # 减少日志：简化资源分配开始日志
        logger.info(f"[{worker_id}] Allocating resources...")
        self.initial_observation = None

        try:
            if not self.active_resources:
                 logger.info(f"[{self.worker_id}] Running in stateless mode (no heavy resources required). Initializing tools only.")
                 # 即使没有需要分配的资源，也调用获取观察值，因为可能有无状态工具可用
                 self.get_inital_obs()
                 return True

            # 1. 申请资源
            # 减少日志：移除批量资源分配详细日志
            # logger.info(f"[{self.worker_id}] Allocating batch resources: {self.active_resources}...")
            res = self._call_tool_sync("allocate_batch_resources", {
                "resource_types": self.active_resources,
                "timeout": 600
            })
            data = self._parse_mcp_response(res)
            if isinstance(data, dict) and data.get("status") == "error":
                 logger.error(f"Alloc failed: {data.get('message')}")
                 return False

            self.allocated_resources = data

            # 2. 初始化资源（总是调用以确保会话同步）
            # 即使没有 resource_init_data，也需要调用 setup_batch_resources 来同步会话
            # 减少日志：移除资源设置日志
            # logger.info(f"[{self.worker_id}] Setting up resources...")
            setup_res = self._call_tool_sync("setup_batch_resources", {
                "resource_init_configs": resource_init_data,  # 可以为空 dict，不影响会话同步
                "allocated_resources": data  # 关键：传递已分配的资源信息用于 _sync_resource_sessions
            })
            setup_result = self._parse_mcp_response(setup_res)
            if setup_result.get("status") not in ["success", "partial_error"]:
                logger.error(f"Setup failed: {setup_result}")
                self.release_resource(self.worker_id)
                return False

            # 3. 获取初始观察
            self.get_inital_obs()
            return True

        except Exception as e:
            logger.error(f"Allocate resource exception: {e}")
            return False

    def release_resource(self, worker_id: str, reset: bool = True) -> None:
        """
        统一释放所有已分配的资源 (MCP 模式)
        调用 system_resource 组的 release_batch_resources 工具
        """
        # 减少日志：简化资源释放开始日志
        logger.info(f"[{worker_id}] Releasing resources...")
        
        # 收集所有已分配资源的 ID
        resource_ids = []
        for res_type, res_data in self.allocated_resources.items():
            if isinstance(res_data, dict) and "id" in res_data:
                resource_ids.append(res_data["id"])
        
        if not resource_ids:
            # 减少日志：移除无资源释放日志
            # logger.info(f"Worker [{worker_id}] has no resources to release.")
            return

        try:
            # [核心修改] 调用 MCP 工具进行批量释放
            self._call_tool_sync("release_batch_resources", {
                "worker_id": worker_id,
                "resource_ids": resource_ids
            })

            # 清空本地记录
            self.allocated_resources.clear()
            # 减少日志：移除释放完成日志
            # logger.info(f"Worker [{worker_id}] release completed.")

        except Exception as e:
            logger.error(f"Failed to release resources via MCP: {e}")

    def get_allocated_resource_id(self) -> str:
        return self.worker_id
    # =========================================================================
    # [新增] 缺失的辅助功能函数 (适配框架调用)
    # =========================================================================

    def get_task_output_dir(self, base_dir: str, task_id: str, model_name: str) -> str:
        """
        生成任务特定的输出目录路径。
        
        Args:
            base_dir: 基础输出目录 (如 'results')
            task_id: 任务 ID
            model_name: 模型名称
            
        Returns:
            完整的输出目录路径 (例如: results/gpt-4/task-001)
        """
        # 对模型名称进行文件系统安全处理
        safe_model = "".join([c if c.isalnum() or c in "-_." else "_" for c in model_name])
        
        # 这里的策略是：将所有日志归档在以 "模型名" 命名的子目录下
        # run_task 中的 _save_conversation_log 会在这个目录下创建 {task_id}.json
        path = os.path.join(base_dir, safe_model)
        
        # 确保目录存在
        os.makedirs(path, exist_ok=True)
        return path

    def initialize_with_task_config(self, task_config: Dict[str, Any]) -> None:
        """
        接受任务级别的特定配置（如分辨率要求、特定环境参数等）。
        框架会在 run_task 之前调用此方法。
        """
        if not task_config:
            return

        # 减少日志：移除任务配置应用日志
        # logger.info(f"[{self.worker_id}] Applying task specific config: {task_config}")
        # 更新实例配置，以便后续 allocate/setup 阶段可以使用新参数
        self.config.update(task_config)

    def init(self):
        """
        Worker 进程启动后的可选初始化钩子。
        框架在实例化环境后会尝试调用此方法。
        """
        # 在 MCP 模式下，连接已经在 env_start 中建立，此处可留空或做额外检查
        pass

    def cleanup(self, worker_id: Optional[str] = None):
        """
        清理资源的统一入口。
        框架在 Worker 退出或收到停止信号时会调用此方法。
        """
        wid = worker_id or self.worker_id
        # 减少日志：移除清理开始日志
        # logger.info(f"[{wid}] Cleaning up environment resources...")
        try:
            # 1. 释放远端资源
            self.release_resource(wid)
            # 2. 关闭 MCP 连接
            self.env_close()
        except Exception as e:
            logger.error(f"[{wid}] Cleanup failed: {e}")

    def _truncate_data(self, data: Any, max_len: int = 100) -> Any:
        """
        辅助函数：递归截断数据结构中的长字符串，仅用于日志展示。
        """
        if isinstance(data, dict):
            return {k: self._truncate_data(v, max_len) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._truncate_data(i, max_len) for i in data]
        elif isinstance(data, str):
            if len(data) > max_len:
                # 保留前 max_len 个字符，并提示总长度
                return f"{data[:max_len]}... [TRUNCATED, total_len={len(data)}]"
            return data
        else:
            return data
