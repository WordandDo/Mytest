"""
Trajectory Sampler

Responsible for sampling trajectory tree starting from seed entity
"""

import openai
import asyncio
import json
import os
import pdb
import bdb
import re
import time
from typing import Dict, List, Any, Optional, Tuple

from models import TrajectoryNode
from synthesis_config import SynthesisConfig

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.http_mcp_env import HttpMCPEnv


class GenericTrajectorySampler:
    """
    Generic Trajectory Sampler supporting arbitrary tool combinations
    """
    
    def __init__(self, 
                 environment: HttpMCPEnv,
                 config: SynthesisConfig):
        """
        Initialize Generic Trajectory Sampler
        """
        self.environment = environment
        self.config = config
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url=os.environ.get("OPENAI_API_URL", os.environ.get("OPENAI_API_BASE", ""))
        )

        # Optional async OpenAI client (used only when async_tree=True)
        self.async_client = None
        try:
            if hasattr(openai, "AsyncOpenAI"):
                self.async_client = openai.AsyncOpenAI(  # type: ignore[attr-defined]
                    api_key=os.environ.get("OPENAI_API_KEY", ""),
                    base_url=os.environ.get("OPENAI_API_URL", os.environ.get("OPENAI_API_BASE", ""))
                )
        except Exception:
            self.async_client = None
        
        # Get available tools information
        self.available_tools = self._get_available_tools()
        self.tool_descriptions = self._generate_tool_descriptions()
        
        # Trajectory tree storage
        self.nodes: Dict[str, TrajectoryNode] = {}
        self.root_id: Optional[str] = None

        # Per-seed action de-duplication (reset in sample_trajectory_tree)
        # Store canonical signatures like: tool_name + sorted(parameters)
        self._seed_used_action_signatures: set = set()
        self._seed_used_action_signatures_ordered: List[str] = []

        # Per-seed node id counter (reset in sample_trajectory_tree)
        self._node_index: int = 1

        # Per-seed timing guard to proactively stop long-running tasks
        self._task_start_time: Optional[float] = None
        self._safe_timeout_s: Optional[float] = None

        # Async runtime primitives (initialized lazily inside async loop)
        self._async_tree_lock: Optional[asyncio.Lock] = None
        self._async_action_lock: Optional[asyncio.Lock] = None
        self._llm_sem: Optional[asyncio.Semaphore] = None
        self._tool_sem: Optional[asyncio.Semaphore] = None
        
    def _get_available_tools(self) -> List[Dict[str, Any]]:
        """Get available tools information from environment"""
        tools = []
        
        # If tools list is specified in config, only use these tools
        if self.config.available_tools:
            tool_names = self.config.available_tools
        else:
            tool_names = self.environment.list_tools()
        
        for tool_name in tool_names:
            tool = self.environment.get_tool(tool_name)
            if tool:
                tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                })
        
        return tools
    
    def _generate_tool_descriptions(self) -> str:
        """Generate tool description text"""
        descriptions = []
        
        for tool in self.available_tools:
            desc = f"\n{len(descriptions) + 1}. {tool['name']}: {tool['description']}\n"
            desc += "   Parameters:\n"
            
            for param in tool['parameters']:
                param_type = param['type']
                if param_type == 'array':
                    param_type = f"array of {param.get('array_type', 'string')}"
                
                required_str = " (required)" if param.get('required', False) else " (optional)"
                desc += f"   - {param['name']} ({param_type}){required_str}: {param['description']}\n"
            
            descriptions.append(desc)
        
        return "\n".join(descriptions)
    
    def sample_trajectory_tree(self, seed_data: str) -> Dict[str, TrajectoryNode]:
        """Sample trajectory tree starting from seed"""
        # IMPORTANT: sampler instance is reused across seeds in pipeline workers,
        # so we must reset per-seed state here.
        self.nodes = {}
        self.root_id = None
        self._seed_used_action_signatures.clear()
        self._seed_used_action_signatures_ordered.clear()
        self._node_index = 1
        self._task_start_time = time.time()
        # Align with pool occupation timeout (default 1800s) and keep a small buffer
        pool_timeout = float(os.environ.get("RESOURCE_MAX_OCCUPATION_TIME", "1800"))
        self._safe_timeout_s = max(60.0, pool_timeout - 100.0)
        # Reset async primitives per-seed to avoid cross-loop binding issues
        self._async_tree_lock = None
        self._async_action_lock = None
        self._llm_sem = None
        self._tool_sem = None

        print(f"\n{'='*60}")
        print(f"Starting Trajectory Tree Sampling")
        print(f"Seed Content: {seed_data}")
        print(f"Environment Mode: {self.environment.mode}")
        print(f"Available Tools: {[t['name'] for t in self.available_tools]}")
        print(f"Max Depth: {self.config.max_depth}, Branching Factor: {self.config.branching_factor}")
        print(f"{'='*60}\n")
        
        # Create root node
        root_id = f"d0_t0_b0"
        observation = f"Starting point: {seed_data}"
        if self.config.seed_description:
            observation = f"Starting point ({self.config.seed_description}): {seed_data}"
        
        root_node = TrajectoryNode(
            node_id=root_id,
            observation=observation,
            intent="Start exploration",
            action=None,
            parent_id=None,
            depth=0
        )
        
        self.nodes[root_id] = root_node
        self.root_id = root_id
        
        # Expand tree (sync by default; async optional)
        use_async = bool(getattr(self.config, "async_tree", False))
        env_supports_async = callable(getattr(self.environment, "execute_tool_async", None)) and callable(getattr(self.environment, "_run_sync", None))
        if use_async and env_supports_async:
            print(f"⚡ Async Tree Expansion Enabled (llm_concurrency={getattr(self.config, 'async_llm_concurrency', 8)}, tool_concurrency={getattr(self.config, 'async_tool_concurrency', 1)})")
            try:
                # Run async expansion inside env loop to keep MCP client/session loop-consistent
                self.environment._run_sync(self._expand_tree_async(root_id, seed_data))  # type: ignore[attr-defined]
            except Exception as e:
                print(f"⚠️  Async expansion failed, falling back to sync: {e}")
                self._expand_tree(root_id, seed_data)
        else:
            # Legacy sync expansion
            self._expand_tree(root_id, seed_data)
        
        print(f"\n✅ Trajectory Tree Sampling Completed!")
        print(f"   Total Nodes: {len(self.nodes)}")
        max_d = max(node.depth for node in self.nodes.values()) if self.nodes else 0
        print(f"   Max Depth: {max_d}")
        
        return self.nodes
    
    def _expand_tree(self, node_id: str, seed_data: str):
        """Recursively expand trajectory tree"""
        if self._check_timeout():
            return

        current_node = self.nodes[node_id]
        
        if current_node.depth >= self.config.max_depth:
            return
        
        print(f"\n🌳 Expanding node {node_id} (depth: {current_node.depth})")
        
        # Dynamically adjust branching factor
        if current_node.depth >= self.config.depth_threshold:
            current_branching_factor = 1
        else:
            current_branching_factor = self.config.branching_factor
        
        for branch_idx in range(current_branching_factor):
            try:
                action, intent = self._generate_next_action(current_node, seed_data)
                
                if action is None:
                    print(f"   Branch {branch_idx + 1}: Unable to generate valid action, skipping")
                    continue
                
                print(f"   Action Generated: {action.get('tool_name')}")
                duplicate_sig = self._recent_duplicate_signature(action, window=5)
                if duplicate_sig:
                    observation = "SYSTEM_BLOCK: You just performed this exact action recently. Do NOT repeat. Try a different query or STOP."
                    print(f"   🔁 Detected recent duplicate action, skipping execution: {duplicate_sig}")
                else:
                    observation = self._execute_action(action)

                # Record used action for this seed (global across the whole exploration tree)
                self._record_used_action(action)
                
                child_depth = current_node.depth + 1
                total_nodes = len(self.nodes)
                child_id = f"d{child_depth}_t{total_nodes}_b{branch_idx}"
                child_node = TrajectoryNode(
                    node_id=child_id,
                    observation=observation,
                    intent=intent,
                    action=action,
                    parent_id=node_id,
                    depth=child_depth
                )
                
                self.nodes[child_id] = child_node
                current_node.children_ids.append(child_id)
                
                print(f"   ✓ Branch {branch_idx + 1}: Created node {child_id}")
                print(f"     Intent: {intent}")
                print(f"     Action: {action.get('tool_name', 'unknown')}")
                # Safe slice for logging (now guaranteed to be string)
                obs_preview = observation[:1000] + "..." if len(observation) > 1000 else observation
                print(f"     Observation: {obs_preview}")
                
                self._expand_tree(child_id, seed_data)
                
            except Exception as e:
                if isinstance(e, bdb.BdbQuit):
                    raise e
                print(f"   ✗ Branch {branch_idx + 1} failed: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

    async def _ensure_async_runtime(self) -> None:
        """
        Initialize async primitives lazily inside the running event loop.
        This avoids cross-event-loop binding issues in multiprocessing workers.
        """
        if self._async_tree_lock is None:
            self._async_tree_lock = asyncio.Lock()
        if self._async_action_lock is None:
            self._async_action_lock = asyncio.Lock()
        if self._llm_sem is None:
            llm_c = int(getattr(self.config, "async_llm_concurrency", 8) or 8)
            self._llm_sem = asyncio.Semaphore(max(1, llm_c))
        if self._tool_sem is None:
            tool_c = int(getattr(self.config, "async_tool_concurrency", 1) or 1)
            self._tool_sem = asyncio.Semaphore(max(1, tool_c))

    async def _expand_tree_async(self, node_id: str, seed_data: str) -> None:
        """Asynchronously expand trajectory tree with bounded concurrency."""
        await self._ensure_async_runtime()
        await self._expand_tree_async_impl(node_id, seed_data)

    async def _expand_tree_async_impl(self, node_id: str, seed_data: str) -> None:
        if self._check_timeout():
            return

        current_node = self.nodes[node_id]

        if current_node.depth >= self.config.max_depth:
            return

        print(f"\n🌳 (async) Expanding node {node_id} (depth: {current_node.depth})")

        # Dynamically adjust branching factor
        if current_node.depth >= self.config.depth_threshold:
            current_branching_factor = 1
        else:
            current_branching_factor = self.config.branching_factor

        branch_tasks = []
        for branch_idx in range(current_branching_factor):
            branch_tasks.append(asyncio.create_task(self._expand_one_branch_async(current_node, node_id, seed_data, branch_idx)))

        results = await asyncio.gather(*branch_tasks, return_exceptions=True)

        child_ids: List[str] = []
        for branch_idx, res in enumerate(results):
            if isinstance(res, Exception):
                print(f"   ✗ (async) Branch {branch_idx + 1} failed: {str(res)}")
                continue
            if isinstance(res, str) and res:
                child_ids.append(res)

        if not child_ids:
            return

        # Expand children concurrently (bounded by semaphores inside LLM/tool calls)
        child_tasks = [asyncio.create_task(self._expand_tree_async_impl(cid, seed_data)) for cid in child_ids]
        await asyncio.gather(*child_tasks, return_exceptions=True)

    async def _expand_one_branch_async(
        self,
        current_node: TrajectoryNode,
        node_id: str,
        seed_data: str,
        branch_idx: int
    ) -> Optional[str]:
        if self._check_timeout():
            return None

        try:
            action, intent = await self._generate_next_action_async(current_node, seed_data)

            if action is None:
                print(f"   Branch {branch_idx + 1}: Unable to generate valid action, skipping")
                return None

            print(f"   Action Generated: {action.get('tool_name')}")
            duplicate_sig = self._recent_duplicate_signature(action, window=5)
            if duplicate_sig:
                observation = "SYSTEM_BLOCK: You just performed this exact action recently. Do NOT repeat. Try a different query or STOP."
                print(f"   🔁 Detected recent duplicate action, skipping execution: {duplicate_sig}")
            else:
                observation = await self._execute_action_async(action)

            # Record used action for this seed (global across the whole exploration tree)
            # (In async mode we may have already reserved it during generation; this is idempotent.)
            self._record_used_action(action)

            child_depth = current_node.depth + 1

            # Create child node (guard shared tree mutation & id allocation)
            assert self._async_tree_lock is not None
            async with self._async_tree_lock:
                child_id = f"d{child_depth}_t{self._node_index}_b{branch_idx}"
                self._node_index += 1

                child_node = TrajectoryNode(
                    node_id=child_id,
                    observation=observation,
                    intent=intent,
                    action=action,
                    parent_id=node_id,
                    depth=child_depth
                )

                self.nodes[child_id] = child_node
                current_node.children_ids.append(child_id)

            print(f"   ✓ Branch {branch_idx + 1}: Created node {child_id}")
            print(f"     Intent: {intent}")
            print(f"     Action: {action.get('tool_name', 'unknown')}")
            obs_preview = observation[:1000] + "..." if len(observation) > 1000 else observation
            print(f"     Observation: {obs_preview}")

            return child_id

        except Exception as e:
            if isinstance(e, bdb.BdbQuit):
                raise e
            print(f"   ✗ Branch {branch_idx + 1} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    async def _chat_completion_create_async(self, **kwargs):
        """Async wrapper for OpenAI chat.completions.create (with sync fallback)."""
        if self.async_client is not None:
            return await self.async_client.chat.completions.create(**kwargs)
        # Fallback: run sync client in a thread (should be rare)
        return await asyncio.to_thread(self.client.chat.completions.create, **kwargs)

    async def _generate_next_action_async(
        self,
        current_node: TrajectoryNode,
        seed_data: str
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """Async version of _generate_next_action with per-seed de-dup lock."""
        history = self._build_history(current_node)
        used_actions_block = self._format_used_actions_for_prompt()
        prompt_base = self._build_action_generation_prompt(
            seed_data=seed_data,
            history=history,
            current_observation=current_node.observation,
            used_actions_block=used_actions_block
        )

        retry = 0
        last_duplicate_sig: Optional[str] = None
        while retry < self.config.max_retries:
            try:
                prompt = prompt_base
                if last_duplicate_sig:
                    prompt += f"""
[Hard Constraint Reminder]
Your previous proposal duplicated an already executed action for this seed and was rejected:
{last_duplicate_sig}
You MUST propose a NEW action different from all items in [Already Explored Actions - Do NOT Repeat].
"""

                assert self._llm_sem is not None
                async with self._llm_sem:
                    response = await self._chat_completion_create_async(
                        model=self.config.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7 + retry * 0.1,
                    )

                content = response.choices[0].message.content
                if not content:
                    raise ValueError("Empty response from LLM")

                cleaned_content = self._clean_json_string(content)
                result = json.loads(cleaned_content)

                if isinstance(result, list):
                    if len(result) > 0 and isinstance(result[0], dict):
                        result = result[0]
                        print("      Notice: LLM returned a list, using first element.")
                    else:
                        raise ValueError("LLM returned an invalid list format.")

                intent = result.get("intent", "")
                action = result.get("action", {})

                if self._validate_action(action):
                    sig = self._action_signature(action)

                    # Atomic check+record to prevent concurrent duplicates
                    assert self._async_action_lock is not None
                    async with self._async_action_lock:
                        if sig in self._seed_used_action_signatures:
                            last_duplicate_sig = sig
                            print(f"      Warning: Duplicate action detected, rejecting (attempt {retry + 1}): {sig}")
                            retry += 1
                            continue
                        # Reserve immediately so other concurrent branches won't choose it
                        self._seed_used_action_signatures.add(sig)
                        self._seed_used_action_signatures_ordered.append(sig)

                    return action, intent

                print(f"      Warning: Invalid action format (attempt {retry + 1})")
                retry += 1

            except Exception as e:
                print(f"      Warning: Failed to generate action (attempt {retry + 1}): {str(e)}")
                retry += 1

        return None, ""

    async def _execute_action_async(self, action: Dict[str, Any]) -> str:
        """Async version of _execute_action (Always returns String)."""
        tool_name = action["tool_name"]
        parameters = action["parameters"]

        try:
            # Prefer true-async env tool execution if available
            if callable(getattr(self.environment, "execute_tool_async", None)):
                assert self._tool_sem is not None
                async with self._tool_sem:
                    result = await self.environment.execute_tool_async(tool_name, parameters)  # type: ignore[func-returns-value]
            else:
                # Fallback to sync tool execution
                result = self.environment.execute_tool(tool_name, parameters)

            if isinstance(result, dict):
                if "text" in result:
                    return str(result["text"])
                return json.dumps(result, ensure_ascii=False)

            return str(result)
        except Exception as e:
            return f"[Error] Action execution failed: {str(e)}"
    
    def _clean_json_string(self, content: str) -> str:
        """
        [新增] 清洗 LLM 返回的字符串，提取有效的 JSON 部分
        """
        # 1. 移除 Markdown 代码块标记 ```json ... ```
        content = re.sub(r'^```json\s*', '', content, flags=re.MULTILINE)
        content = re.sub(r'^```\s*', '', content, flags=re.MULTILINE)
        content = re.sub(r'\s*```$', '', content, flags=re.MULTILINE)
        
        # 2. 尝试提取第一个 { ... } 或 [ ... ]
        # 这能解决 "Here is the JSON: {...}" 这种前缀问题
        try:
            # 寻找第一个左大括号或左中括号
            match = re.search(r'(\{.*\}|\[.*\])', content, re.DOTALL)
            if match:
                return match.group(1)
        except:
            pass
            
        return content.strip()

    def _generate_next_action(self, 
                              current_node: TrajectoryNode, 
                              seed_data: str) -> Tuple[Optional[Dict[str, Any]], str]:
        """Generate next action and intent based on current state"""
        history = self._build_history(current_node)
        used_actions_block = self._format_used_actions_for_prompt()
        prompt_base = self._build_action_generation_prompt(
            seed_data=seed_data,
            history=history,
            current_observation=current_node.observation,
            used_actions_block=used_actions_block
        )
        
        retry = 0
        last_duplicate_sig: Optional[str] = None
        while retry < self.config.max_retries:
            try:
                prompt = prompt_base
                if last_duplicate_sig:
                    prompt += f"""
[Hard Constraint Reminder]
Your previous proposal duplicated an already executed action for this seed and was rejected:
{last_duplicate_sig}
You MUST propose a NEW action different from all items in [Already Explored Actions - Do NOT Repeat].
"""
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7 + retry * 0.1,
                    # 某些 OSS 模型不支持 response_format="json_object"，如果报错可尝试移除此行
                    #response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content
                
                if not content:
                    raise ValueError("Empty response from LLM")
                
                # [关键修复] 清洗并解析 JSON
                cleaned_content = self._clean_json_string(content)
                result = json.loads(cleaned_content)

                # print(result)
                
                # [关键修复] 处理返回列表的情况 (修复 'list' object has no attribute 'get')
                if isinstance(result, list):
                    if len(result) > 0 and isinstance(result[0], dict):
                        result = result[0]
                        print("      Notice: LLM returned a list, using first element.")
                    else:
                        raise ValueError("LLM returned an invalid list format.")
                
                intent = result.get("intent", "")
                action = result.get("action", {})

                # print(intent)
                # print(action)
                # print(self.available_tools)

                if self._validate_action(action):
                    # Enforce per-seed global action de-duplication (tool_name + parameters)
                    sig = self._action_signature(action)
                    if sig in self._seed_used_action_signatures:
                        last_duplicate_sig = sig
                        print(f"      Warning: Duplicate action detected, rejecting (attempt {retry + 1}): {sig}")
                        retry += 1
                        continue
                    return action, intent
                
                print(f"      Warning: Invalid action format (attempt {retry + 1})")
                retry += 1
                
            except Exception as e:
                print(f"      Warning: Failed to generate action (attempt {retry + 1}): {str(e)}")
                retry += 1
        
        return None, ""
    
    def _build_action_generation_prompt(
        self,
        seed_data: str,
        history: str,
        current_observation: str,
        used_actions_block: str = ""
    ) -> str:
        """Build action generation prompt"""
        
        system_instruction = ""
        if hasattr(self.environment, "get_system_prompt"):
            system_instruction = self.environment.get_system_prompt(task_question=f"Exploration Task: {seed_data}")
        else:
            system_instruction = "You are an intelligent Agent using available tools for exploration and reasoning."
        
        prompt = f"""{system_instruction}

[Starting Point Information]
Content: {seed_data}"""
        
        if self.config.seed_description:
            prompt += f"\nDescription: {self.config.seed_description}"
        
        prompt += """

[Exploration Goal]
Based on the starting point content and available tools, conduct systematic exploration to collect and reason about valuable information.
Finally, I will synthesize a question and answer based on your collected information. Therefore, you should explore sufficient information for me.
"""
        if used_actions_block:
            prompt += f"""
[Already Explored Actions - Do NOT Repeat]
The following tool calls (tool_name + parameters) have ALREADY been executed for this seed.
You MUST propose a NEW action that is NOT in this list or similar to them to increase the diversity of the exploration. Repeating any of them is strictly forbidden.
{used_actions_block}
"""
        
        if self.config.sampling_tips:
            prompt += f"""[Exploration Strategy and Focus]
{self.config.sampling_tips}

"""
        
        prompt += f"""Current History Trajectory:
{history}

Current Observation:
{current_observation}

Available Tools:
{self.tool_descriptions}

"""
        
#         if self.config.qa_examples:
#             prompt += """Reference Examples:\n"""
#             for i, example in enumerate(self.config.qa_examples[:2], 1):
#                 prompt += f"""Example {i}:
# Question: {example.get('question', '')}
# Answer: {example.get('answer', '')}
# """
        
        prompt += """
Based on the current state and available tools, select an appropriate tool and parameters, and generate the next action and intent.

IMPORTANT: Return ONLY a valid JSON object. Do not wrap it in markdown blocks.
Format:
{
    "intent": "The intent and reason for executing this action",
    "action": {
        "tool_name": "tool name",
        "parameters": {parameter dictionary}
    }
}
"""
        return prompt

    def _action_signature(self, action: Dict[str, Any]) -> str:
        """
        Canonical signature for action de-duplication within a seed.
        Format: tool_name + canonicalized JSON(parameters)
        """
        tool_name = ""
        parameters: Any = {}
        try:
            tool_name = str(action.get("tool_name", ""))
            parameters = action.get("parameters", {}) if isinstance(action, dict) else {}
        except Exception:
            tool_name = ""
            parameters = {}

        # Best-effort canonicalization (stable key order)
        try:
            params_str = json.dumps(parameters, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        except Exception:
            params_str = str(parameters)

        return f"{tool_name}({params_str})"

    def _record_used_action(self, action: Dict[str, Any]) -> None:
        """Record an executed action for this seed (global across the trajectory tree)."""
        sig = self._action_signature(action)
        if sig not in self._seed_used_action_signatures:
            self._seed_used_action_signatures.add(sig)
            self._seed_used_action_signatures_ordered.append(sig)

    def _format_used_actions_for_prompt(self, max_items: int = 40, max_line_chars: int = 300) -> str:
        """Format executed actions list for inclusion in prompt (truncate to keep prompt bounded)."""
        if not self._seed_used_action_signatures_ordered:
            return "None yet."

        # Use the most recent actions (often most relevant for avoiding repetition)
        items = self._seed_used_action_signatures_ordered[-max_items:]
        lines = []
        for i, sig in enumerate(items, 1):
            s = sig
            if len(s) > max_line_chars:
                s = s[:max_line_chars] + "...(truncated)"
            lines.append(f"- {i}. {s}")

        omitted = len(self._seed_used_action_signatures_ordered) - len(items)
        if omitted > 0:
            lines.insert(0, f"(Showing last {len(items)} actions; {omitted} earlier actions omitted but still forbidden.)")

        return "\n".join(lines)
    
    def _validate_action(self, action: Dict[str, Any]) -> bool:
        """Validate whether action format is correct"""
        if not isinstance(action, dict):
            return False
        
        tool_name = action.get("tool_name", "")
        parameters = action.get("parameters", {})
        
        available_tool_names = [t['name'] for t in self.available_tools]
        if tool_name not in available_tool_names:
            return False
        
        # Find tool definition
        tool_def = None
        for t in self.available_tools:
            if t['name'] == tool_name:
                tool_def = t
                break
        
        if not tool_def:
            return False
        
        # Validate required parameters
        for param in tool_def['parameters']:
            if param.get('required', False):
                if param['name'] not in parameters:
                    return False

        # Normalize parameter types (e.g., numeric strings -> int) to be more tolerant of LLM output
        normalized_params = self._normalize_parameters(tool_def, parameters)
        action["parameters"] = normalized_params
        
        return True

    def _normalize_parameters(self, tool_def: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize common parameter types to increase robustness against LLM format issues.
        Example: convert numeric strings to int/float when the tool expects a number.
        """
        normalized = dict(parameters)

        for param in tool_def.get("parameters", []):
            name = param.get("name")
            expected_type = param.get("type")
            if name is None or name not in normalized:
                continue

            value = normalized[name]

            # Convert numeric strings to int/float
            if isinstance(value, str) and expected_type in ("integer", "number"):
                try:
                    normalized[name] = int(value) if expected_type == "integer" else float(value)
                except ValueError:
                    # If conversion fails, leave original value
                    pass

        return normalized
    
    def _execute_action(self, action: Dict[str, Any]) -> str:
        """Execute action and return observation (Always returns String)"""
        tool_name = action["tool_name"]
        parameters = action["parameters"]
        
        try:
            result = self.environment.execute_tool(tool_name, parameters)
            
            # 兼容 MCP 环境的字典格式 {'text': ..., 'images': ...}
            if isinstance(result, dict):
                # 1. 优先提取文本，忽略图片
                if "text" in result:
                    return str(result["text"])
                # 2. 如果没有text字段，序列化整个字典
                return json.dumps(result, ensure_ascii=False)
                
            return str(result)
        except Exception as e:
            return f"[Error] Action execution failed: {str(e)}"

    def _check_timeout(self) -> bool:
        """Return True if the current seed run exceeded the safe timeout."""
        if self._task_start_time is None or self._safe_timeout_s is None:
            return False
        elapsed = time.time() - self._task_start_time
        if elapsed > self._safe_timeout_s:
            print(f"⏳ Safe timeout reached ({elapsed:.1f}s > {self._safe_timeout_s:.1f}s). Stopping expansion to avoid zombie state.")
            return True
        return False

    def _recent_duplicate_signature(self, action: Dict[str, Any], window: int = 5) -> Optional[str]:
        """Detect if action matches any of the recent executed actions."""
        sig = self._action_signature(action)
        recent = self._seed_used_action_signatures_ordered[-window:]
        if sig in recent:
            return sig
        return None
    
    def _build_history(self, node: TrajectoryNode) -> str:
        """Build history trajectory from root to current node"""
        path = []
        current = node
        
        while current.parent_id is not None:
            path.append(current)
            current = self.nodes[current.parent_id]
        
        path.reverse()
        
        if not path:
            return "No history trajectory"
        
        history_str = ""
        for i, n in enumerate(path, 1):
            history_str += f"\nStep {i}:\n"
            history_str += f"  Intent: {n.intent}\n"
            if n.action:
                history_str += f"  Action: {n.action.get('tool_name', 'unknown')}\n"
            
            # Ensure observation is treated as string for slicing
            obs_str = str(n.observation)
            history_str += f"  Observation: {obs_str[:1000]}...\n"
        
        return history_str
