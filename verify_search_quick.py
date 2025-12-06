import logging
import sys
import os
from src.envs.factory import create_environment

# 配置日志到控制台
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SearchVerify")

def main():
    # 1. 创建环境
    logger.info("🛠️  Initializing HttpMCPSearchEnv...")
    try:
        env = create_environment(
            mode="http_mcp_search",
            model_name="gpt-4o",
            # 确保传递了正确的 Gateway 地址
            mcp_server_url="http://localhost:8080", 
            gateway_config_path="gateway_config.json"
        )
        env.env_start()
        logger.info("✅ Environment started and connected to Gateway.")
    except Exception as e:
        logger.error(f"❌ Failed to init environment: {e}")
        return

    # 2. 定义测试任务
    task = {
        "id": "verify_001",
        "question": "请搜索 'OpenAI GPT-4o' 的发布日期，并简述其相比 GPT-4 的主要改进点。"
    }

    # 3. 运行任务
    agent_config = {
        "model_name": "gpt-4o",
        "max_turns": 5,
        "task_timeout": 60
    }
    
    logger.info(f"🚀 Running task: {task['question']}")
    try:
        result = env.run_task(task, agent_config, logger)
        
        print("\n" + "="*50)
        print(f"📊 Result Success: {result['success']}")
        print("-" * 20)
        print(f"📝 Final Answer: {result['answer']}")
        print("-" * 20)
        print("🔧 Tool Calls:")
        for msg in result['messages']:
            if msg.get('role') == 'assistant' and msg.get('tool_calls'):
                for tc in msg['tool_calls']:
                    print(f"   -> {tc['function']['name']}")
        print("="*50 + "\n")
        
    except Exception as e:
        logger.error(f"❌ Execution failed: {e}")
    finally:
        env.cleanup()
        logger.info("🧹 Environment cleaned up.")

if __name__ == "__main__":
    main()