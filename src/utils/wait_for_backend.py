import time
import json
import os
import sys
import requests
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [Wait] - %(message)s')
logger = logging.getLogger("WaitForBackend")

def load_enabled_resources(config_path="deployment_config.json"):
    """读取配置文件，返回所有被启用的资源名称列表"""
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return []

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        enabled_resources = []
        resources = config.get("resources", {})
        
        for name, settings in resources.items():
            # 检查 enabled 字段
            if settings.get("enabled", False):
                enabled_resources.append(name)
        
        return enabled_resources
    except Exception as e:
        logger.error(f"Failed to parse config: {e}")
        return []

def check_backend_status(url, expected_resources):
    """检查后端状态，返回 (是否就绪, 缺失的资源列表)"""
    try:
        resp = requests.get(f"{url}/status", timeout=2)
        if resp.status_code != 200:
            return False, f"HTTP {resp.status_code}"
        
        status_data = resp.json()
        
        # 检查所有期望的资源是否都在 status_data 中
        missing = []
        for resource in expected_resources:
            # 这里的 status_data key 通常是资源名 (如 'rag', 'vm_computer_13')
            if resource not in status_data:
                missing.append(resource)
            else:
                # 进一步检查资源池是否真的初始化了 (total > 0)
                pool_stats = status_data[resource]
                if pool_stats.get("total", 0) <= 0:
                    missing.append(f"{resource}(initializing)")
        
        if missing:
            return False, missing
        
        return True, None

    except requests.exceptions.ConnectionError:
        return False, "Connection refused (Service down)"
    except Exception as e:
        return False, str(e)

def main():
    api_url = os.environ.get("RESOURCE_API_URL", "http://localhost:8000")
    config_path = "deployment_config.json"
    
    # 1. 确定需要等待哪些资源
    expected_resources = load_enabled_resources(config_path)
    logger.info(f"Target resources to wait for: {expected_resources}")
    
    if not expected_resources:
        logger.warning("No enabled resources found in config. Exiting.")
        return

    # 2. 循环等待
    start_time = time.time()
    max_wait = 900  # 最大等待 15 分钟
    
    logger.info(f"Waiting for backend at {api_url}...")
    
    while True:
        if time.time() - start_time > max_wait:
            logger.error("Timeout waiting for backend resources.")
            sys.exit(1)

        is_ready, reason = check_backend_status(api_url, expected_resources)
        
        if is_ready:
            logger.info("✅ All backend resources are ready!")
            break
        else:
            # 打印未就绪的原因（例如：缺少 rag, 或连接被拒绝）
            if isinstance(reason, list):
                logger.info(f"⏳ Waiting for resources: {reason}")
            else:
                logger.info(f"⏳ Backend not ready: {reason}")
            
            time.sleep(5)

if __name__ == "__main__":
    main()