import asyncio
import json
import os
import sys
import argparse
import logging
from typing import List, Dict, Any
from datetime import datetime

# [新增] 引入 dotenv 自动加载 .env 文件
from dotenv import load_dotenv

# 确保能引用项目根目录
sys.path.append(os.getcwd())

# 引入 OpenAI SDK
from openai import AsyncOpenAI

# 引入我们的 MCP 环境
from src.envs.mcp_env import OSWorldMCPEnv

# --- 配置日志 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("RolloutRunner")

# [新增] 在脚本启动时立即加载环境变量
load_dotenv()

class SimpleAgent:
    """
    一个简单的 ReAct Agent，用于驱动 MCP 环境
    """
    def __init__(self, model: str, client: AsyncOpenAI):
        self.model = model
        self.client = client

    async def solve_task(self, env: OSWorldMCPEnv, task: Dict) -> Dict:
        task_id = task['id']
        logger.info(f"[{task_id}] Agent started. Goal: {task['instruction']}")
        
        # 1. 环境初始化 (Setup)
        try:
            logger.info(f"[{task_id}] Requesting environment setup...")
            init_obs = await env.setup_task(
                config_name=task.get("config", "standard"), 
                task_id=task_id
            )
            logger.info(f"[{task_id}] Setup complete. Observation received.")
        except Exception as e:
            logger.error(f"[{task_id}] Setup failed: {e}")
            return {"status": "setup_failed", "error": str(e)}

        # 2. 构造对话历史 (包含初始 Vision 消息)
        # 检查初始观察中是否有截图，如果有，分离出来构建 Vision 消息
        init_screenshot = None
        init_obs_text = init_obs
        
        if isinstance(init_obs, dict) and "screenshot" in init_obs and init_obs["screenshot"]:
            init_obs_text = init_obs.copy()
            init_screenshot = init_obs_text.pop("screenshot")
            init_obs_text["screenshot"] = "<screenshot_in_vision_block>"

        # 构建 System Message
        messages = [
            {"role": "system", "content": "You are an expert in computer automation. Use the provided tools to complete the task."}
        ]

        # 构建 User Message (Initial State)
        user_content = [
            {"type": "text", "text": f"Goal: {task['instruction']}\nEnvironment Ready.\nInitial State:\n{json.dumps(init_obs_text)}"}
        ]
        
        if init_screenshot:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{init_screenshot}"}
            })
            
        messages.append({"role": "user", "content": user_content})

        # 3. 思考-执行循环 (Think-Act Loop)
        max_turns = 15 
        for i in range(max_turns):
            logger.info(f"[{task_id}] Turn {i+1}: Thinking...")
            
            # (A) LLM 推理
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=env.tools_schema, 
                    tool_choice="auto"
                )
            except Exception as e:
                logger.error(f"[{task_id}] LLM API Error: {e}")
                break

            msg = response.choices[0].message
            messages.append(msg)

            # (B) 如果 LLM 没有调用工具，说明任务可能结束了
            if not msg.tool_calls:
                logger.info(f"[{task_id}] Agent returned final answer: {msg.content}")
                break

            # (C) 执行工具
            for tool_call in msg.tool_calls:
                func_name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                except:
                    args = {}
                
                logger.info(f"[{task_id}] >>> Executing Tool: {func_name}({args})")
                
                # 调用 MCP 环境
                obs = await env.step(func_name, args)
                
                logger.info(f"[{task_id}] <<< Tool Output Received (Size: {len(str(obs))} chars)")

                # =================================================================
                # [核心修改] 构建 Vision 格式消息
                # 1. Tool Message: 仅包含 JSON 文本（移除截图）
                # 2. User Message: 包含 Base64 截图 (image_url)
                # =================================================================
                
                screenshot_b64 = None
                obs_for_text = obs
                
                if isinstance(obs, dict):
                    obs_for_text = obs.copy() # 浅拷贝
                    # 提取并移除截图，防止文本 Payload 爆炸
                    if "screenshot" in obs_for_text and obs_for_text["screenshot"]:
                        screenshot_b64 = obs_for_text.pop("screenshot")
                        obs_for_text["screenshot"] = "<screenshot_moved_to_next_message>"
                
                text_content = json.dumps(obs_for_text)

                # 1. 添加 Tool Message (必须有，用于闭合 tool_call)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": func_name,
                    "content": text_content
                })

                # 2. 添加额外的 User Message (展示视觉信息)
                if screenshot_b64:
                    vision_message = {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": "Here is the screen observation after the action:"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{screenshot_b64}"
                                    # 可选: "detail": "high"
                                }
                            },
                            {
                                # 同时附上 Accessibility Tree 文本辅助理解
                                "type": "text",
                                "text": f"Accessibility Info:\n{obs_for_text.get('accessibility_tree', '')}"
                            }
                        ]
                    }
                    messages.append(vision_message)
                    logger.info(f"[{task_id}] Added Vision message with screenshot.")
                
                # =================================================================
                # [结束修改]
                # =================================================================

        # 4. 评估 (Evaluate)
        logger.info(f"[{task_id}] Evaluating task performance...")
        score = await env.evaluate()
        logger.info(f"[{task_id}] Final Score: {score}")

        return {"task_id": task_id, "score": score}

async def worker(sem: asyncio.Semaphore, task: Dict, client: AsyncOpenAI, server_script: str):
    task_id = task['id']
    async with sem:
        logger.info(f"[{task_id}] Worker spawned. Initializing MCP Env...")
        env = OSWorldMCPEnv(server_script=server_script, env_id=task_id)
        try:
            await env.connect()
            agent = SimpleAgent(args.model, client) # 使用全局 args.model
            result = await agent.solve_task(env, task)
            return result
        except Exception as e:
            logger.error(f"[{task_id}] Critical Error: {e}", exc_info=True)
            return {"task_id": task_id, "error": str(e)}
        finally:
            logger.info(f"[{task_id}] Teardown environment...")
            await env.teardown()

async def main():
    global args
    parser = argparse.ArgumentParser(description="Run OSWorld MCP Rollout")
    parser.add_argument("--data", type=str, required=True, help="Path to task JSON file")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--model", type=str, default="gpt-4-turbo", help="OpenAI Model name")
    parser.add_argument("--server_script", type=str, default="src/mcp_server/main.py", help="Path to MCP Server script")
    
    args = parser.parse_args()

    if not os.path.exists(args.data):
        logger.error(f"Data file not found: {args.data}")
        return

    # 读取任务
    with open(args.data, "r") as f:
        tasks = json.load(f)
        if isinstance(tasks, dict): tasks = [tasks]
    
    logger.info(f"Loaded {len(tasks)} tasks. Concurrency: {args.concurrency}")

    # 初始化 OpenAI 客户端
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")

    if not api_key:
        logger.error("❌ OPENAI_API_KEY not found in .env or environment variables!")
        return
    
    logger.info(f"Initializing OpenAI Client (Base URL: {base_url or 'Default'})")
    
    aclient = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url
    )

    sem = asyncio.Semaphore(args.concurrency)
    futures = []
    
    start_time = datetime.now()
    for task in tasks:
        futures.append(worker(sem, task, aclient, args.server_script))
    
    results = await asyncio.gather(*futures)
    duration = datetime.now() - start_time
    
    logger.info(f"All tasks completed in {duration}. Results:")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user.")