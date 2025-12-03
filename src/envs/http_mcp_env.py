# src/envs/http_mcp_env.py
import sys
import os
import json
import logging
import asyncio
import time
from typing import Dict, Any, Union, Optional, List, Tuple
from datetime import datetime

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

        # 初始化持久事件循环
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        logger.info(f"HttpMCPEnv initialized for {self.worker_id} -> {self.server_url}")
        logger.info(f"Active Allocatable Resources: {self.active_resources}")
        
        # 初始化远程工具列表
        self._initialize_tools()

    @property
    def mode(self) -> str:
        return "http_mcp"

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

        messages.append({"role": "user", "content": user_content})

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
                    logger.info(f"Turn {turn_count}: Calling LLM...")
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

                            logger.info(f"Round {turn_count}: 🔧 Using tool: {tool_name}")

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

            logger.info(f"[{self.worker_id}] Initializing OpenAI client...")

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
            return {"modules": [{"resource_type": "vm"}]}
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {"modules": [{"resource_type": "vm"}]}

    def _initialize_tools(self):
        """从 MCP Server 获取工具列表"""
        if not self._tools_initialized:
            return

        try:
            logger.info(f"[{self.worker_id}] Fetching tools from MCP Server...")
            mcp_tools = self._list_tools_sync()

            # 黑名单：Agent 不应直接调用的系统工具
            blacklist = {
                "get_observation", "evaluate_task",
                "allocate_batch_resources", "setup_batch_resources",
                "get_batch_initial_observations", "setup_vm_session",
                "setup_rag_session", "teardown_environment", "release_rag_session",
            }

            valid_tools = [t for t in mcp_tools if t.name not in blacklist]
            self.tool_schemas = [self._convert_mcp_tool_to_openai(t) for t in valid_tools]

            descriptions = [f"- {t.name}: {t.description or 'No description.'}" for t in valid_tools]
            self.tool_descriptions = "\n".join(descriptions)

            logger.info(f"[{self.worker_id}] Initialized {len(valid_tools)} tools.")

        except Exception as e:
            logger.error(f"Failed to initialize tools: {e}")
            self.tool_schemas = []
            self.tool_descriptions = "Error loading tools."

    def _convert_mcp_tool_to_openai(self, mcp_tool) -> ChatCompletionToolParam:
        parameters = mcp_tool.inputSchema.copy() if hasattr(mcp_tool, "inputSchema") else {}
        # 移除内部参数 worker_id
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

    def _call_tool_sync(self, name, arguments) -> Union[Dict[str, Any], Any]:
        """同步调用 MCP 工具"""
        if isinstance(arguments, dict) and "worker_id" not in arguments:
            arguments["worker_id"] = self.worker_id
            
        logger.info(f"[{self.worker_id}] ⏳ Calling tool: {name}")
        res = self._run_sync(self.mcp_client.call_tool(name, arguments))
        
        # 生命周期工具直接返回原始结果，不进行标准化包装
        lifecycle_tools = {
            "allocate_batch_resources", "setup_batch_resources", 
            "get_batch_initial_observations", "teardown_environment"
        }
        if name in lifecycle_tools:
            return res 

        # 通用输出标准化
        standardized_output = {"text": "", "images": []}
        text_parts = []
        
        if hasattr(res, 'content') and res.content:
            for item in res.content:
                if item.type == 'text':
                    text_parts.append(item.text)
                elif item.type == 'image':
                    standardized_output["images"].append(item.data)
                elif item.type == 'resource':
                    text_parts.append(f"[Resource: {item.resource.uri}]")
        else:
            text_parts.append(str(res) if res else "Success")

        standardized_output["text"] = "\n".join(text_parts)
        return standardized_output

    def _parse_mcp_response(self, response: CallToolResult) -> Dict[str, Any]:
        try:
            if response.content and len(response.content) > 0:
                content_item = response.content[0]
                text_content = getattr(content_item, 'text', None)
                if not text_content and hasattr(content_item, 'resource'):
                    text_content = getattr(content_item.resource, 'text', None)
                
                if text_content:
                    return json.loads(text_content)
            return {"status": "unknown"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_inital_obs(self) -> Dict[str, Any]:
        """调用 MCP 获取初始观察"""
        logger.info(f"[{self.worker_id}] Fetching initial observations...")
        combined_obs = {"vm": None, "rag": None}
        try:
            res = self._call_tool_sync("get_batch_initial_observations", {"worker_id": self.worker_id})
            data = self._parse_mcp_response(res)

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
            logger.error(f"[{self.worker_id}] Obs fetch error: {e}")
            return combined_obs

    def allocate_resource(self, worker_id: str, resource_init_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        统一的资源分配入口函数 (MCP 模式)
        """
        resource_init_data = resource_init_data or {}
        logger.info(f"Worker [{worker_id}] allocating resources...")
        self.initial_observation = None

        try:
            if not self.active_resources:
                 logger.warning(f"[{worker_id}] No allocatable resources found (system filtered). Skipping allocation.")
                 # 即使没有需要分配的资源，也调用获取观察值，因为可能有无状态工具可用
                 self.get_inital_obs()
                 return True

            # 1. 申请资源
            logger.info(f"[{self.worker_id}] Allocating batch resources: {self.active_resources}...")
            res = self._call_tool_sync("allocate_batch_resources", {
                "resource_types": self.active_resources,
                "timeout": 600
            })
            data = self._parse_mcp_response(res)
            if isinstance(data, dict) and data.get("status") == "error":
                 logger.error(f"Alloc failed: {data.get('message')}")
                 return False

            self.allocated_resources = data

            # 2. 初始化资源
            if resource_init_data:
                logger.info(f"[{self.worker_id}] Setting up resources...")
                setup_res = self._call_tool_sync("setup_batch_resources", {
                    "resource_init_configs": resource_init_data,
                    "allocated_resources": data
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
        logger.info(f"Worker [{worker_id}] releasing resources via MCP...")
        
        # 收集所有已分配资源的 ID
        resource_ids = []
        for res_type, res_data in self.allocated_resources.items():
            if isinstance(res_data, dict) and "id" in res_data:
                resource_ids.append(res_data["id"])
        
        if not resource_ids:
            logger.info(f"Worker [{worker_id}] has no resources to release.")
            return

        try:
            # [核心修改] 调用 MCP 工具进行批量释放
            self._call_tool_sync("release_batch_resources", {
                "worker_id": worker_id,
                "resource_ids": resource_ids
            })
            
            # 清空本地记录
            self.allocated_resources.clear()
            logger.info(f"Worker [{worker_id}] release completed.")
            
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
        
        logger.info(f"[{self.worker_id}] Applying task specific config: {task_config}")
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
        logger.info(f"[{wid}] Cleaning up environment resources...")
        try:
            # 1. 释放远端资源
            self.release_resource(wid)
            # 2. 关闭 MCP 连接
            self.env_close()
        except Exception as e:
            logger.error(f"[{wid}] Cleanup failed: {e}")