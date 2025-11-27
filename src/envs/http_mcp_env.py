# src/envs/http_mcp_env.py
import time
import random
import requests
import json
import logging
from typing import Dict, Any, Union
from envs.enviroment import Environment

logger = logging.getLogger(__name__)

class HttpMCPEnv(Environment):
    """
    HTTP MCP 环境客户端。
    支持 N-Worker 自动排队机制，连接 SSE 模式的 MCP Server。
    """
    def __init__(self, resource_manager=None, parallel_degree=1, **kwargs):
        # 1. 配置
        self.server_url = kwargs.get("mcp_server_url", "http://localhost:8080")
        self.resource_api_url = kwargs.get("resource_api_url", "http://localhost:8000")
        
        # 2. 获取 worker_id (必需)
        # 优先从 kwargs 获取，否则尝试从进程名获取 (适配 run_parallel_rollout)
        if "worker_id" in kwargs:
            self.worker_id = kwargs["worker_id"]
        else:
            import multiprocessing
            self.worker_id = multiprocessing.current_process().name
            
        logger.info(f"HttpMCPEnv initialized for {self.worker_id} -> {self.server_url}")
        super().__init__(**kwargs)

    @property
    def mode(self) -> str:
        return "http_mcp"

    def _initialize_tools(self):
        # 暂不动态获取 Schema，假设使用 computer_13 标准工具集
        # 如需动态获取，可访问 Server 的 schema 端点
        pass

    def _call_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """发送 HTTP POST 请求调用工具"""
        payload = args.copy()
        payload["worker_id"] = self.worker_id  # 自动注入 ID

        # FastMCP 默认工具路径
        url = f"{self.server_url}/tools/{tool_name}"
        
        try:
            # 使用 HTTP POST
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            try:
                return resp.json()
            except:
                return resp.text
        except Exception as e:
            logger.error(f"MCP Call {tool_name} failed: {e}")
            raise

    def env_start(self):
        """
        [智能排队启动]
        先检查资源池状态，有空闲再申请；无空闲则休眠。
        """
        logger.info(f"Worker [{self.worker_id}] starting setup...")
        retry_count = 0
        while True:
            try:
                # 1. 探路 (Smart Check)
                try:
                    status_resp = requests.get(f"{self.resource_api_url}/status", timeout=5)
                    if status_resp.status_code == 200:
                        stats = status_resp.json()
                        free = stats.get("free", 0)
                        if free <= 0:
                            wait_time = random.uniform(10, 20)
                            logger.info(f"Pool full (0 free). Sleeping {wait_time:.1f}s...")
                            time.sleep(wait_time)
                            continue # 继续循环等待
                except Exception:
                    # 如果查不到状态，直接尝试申请
                    pass

                # 2. 申请 (Allocate)
                # [修改] 只有在真正尝试申请时才打印日志，减少刷屏
                if retry_count == 1 or retry_count % 5 == 0:
                    logger.info(f"Worker [{self.worker_id}] attempting allocation (Attempt {retry_count})...")
                res = self._call_tool("setup_environment", {
                    "config_name": "default",
                    "task_id": "init_check"
                })
                
                # 检查逻辑错误 (MCP 返回 200 但内容是 error)
                if isinstance(res, str):
                    try: res_json = json.loads(res)
                    except: res_json = {}
                else:
                    res_json = res

                if isinstance(res_json, dict) and res_json.get("status") == "error":
                    msg = res_json.get("message", "")
                    if "No resources" in msg or "exhausted" in msg:
                        raise Exception("Resource busy")
                    else:
                        raise RuntimeError(f"Setup error: {msg}")

                logger.info(f"Worker [{self.worker_id}] setup success!")
                break

            except Exception as e:
                # 捕获 HTTP 503 或自定义 busy 异常
                if "503" in str(e) or "Resource busy" in str(e):
                    time.sleep(random.uniform(5, 15))
                else:
                    logger.error(f"Setup failed fatal: {e}")
                    raise e

    def execute_tool(self, tool_name: str, params: Union[str, dict], **kwargs) -> Any:
        if isinstance(params, str):
            try: params = json.loads(params)
            except: params = {"arg": params}
        
        result = self._call_tool(tool_name, params)
        if isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False)
        return str(result)

    def env_close(self):
        try:
            self._call_tool("teardown_environment", {})
        except:
            pass