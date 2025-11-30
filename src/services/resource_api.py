# src/services/resource_api.py
import sys
import os
import json
import re  # [新增]
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any  # 确保导入 Optional
import logging
import uvicorn

cwd = os.getcwd()
sys.path.append(cwd)
sys.path.append(os.path.join(cwd, "src"))

# [修改] 导入新的管理器类
from services.simple_manager import GenericResourceManager

# 1. 加载 .env 到环境变量
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ResourceAPI")

app = FastAPI()
# [修改] 类型注解更新
manager: Optional[GenericResourceManager] = None

# [新增] 带有环境变量替换功能的配置加载器
def load_deployment_config(path: str = "deployment_config.json") -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    
    logger.info(f"Loading config from {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 正则替换 ${VAR_NAME}
    def replace_env(match):
        var_name = match.group(1)
        value = os.environ.get(var_name)
        if value is None:
            # 你可以选择报错，或者保留原样，这里选择警告并保留空字符串以防崩溃
            logger.warning(f"⚠️ Environment variable {var_name} not found in .env")
            return "" 
        return value

    # 执行替换
    content_with_env = re.sub(r'\$\{(\w+)\}', replace_env, content)
    
    try:
        return json.loads(content_with_env)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON config after env substitution: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    global manager
    try:
        # 2. 加载统一配置
        config = load_deployment_config("deployment_config.json")
        
        # 3. 初始化通用管理器
        manager = GenericResourceManager(config)
        
        loop = asyncio.get_running_loop()
        success = await loop.run_in_executor(None, manager.initialize)
        
        if not success:
            logger.error("Failed to start Resource Manager (some pools may be offline)!")
        
        asyncio.create_task(monitor_resource_usage())
    except Exception as e:
        logger.error(f"Critical startup error: {e}", exc_info=True)
        sys.exit(1)

async def monitor_resource_usage():
    logger.info("Starting resource usage monitor (interval=30s)...")
    while True:
        try:
            if manager:
                stats = manager.get_status()
                # [修改] 动态打印所有资源池状态
                log_parts = ["📊 [Monitor]"]
                for name, s in stats.items():
                    log_parts.append(f"{name.upper()}(Free:{s.get('free')}/{s.get('total')})")
                logger.info(" ".join(log_parts))
        except Exception as e:
            logger.error(f"Monitor error: {e}")
        await asyncio.sleep(30)

# [修改] 请求模型增加 resource_type
class AllocReq(BaseModel):
    worker_id: str
    timeout: float = 60.0
    type: str = "vm"  # 默认为 vm，兼容旧代码

class ReleaseReq(BaseModel):
    resource_id: str
    worker_id: str

# [修改] 将top_k改为Optional，默认为None，表示"使用服务器配置的默认值"
class RAGQueryReq(BaseModel):
    resource_id: str
    worker_id: str
    query: str
    # [修改] 改为 Optional，默认为 None，表示"使用服务器配置的默认值"
    top_k: Optional[int] = None

@app.post("/allocate")
def allocate_resource(req: AllocReq):
    try:
        # GenericResourceManager.allocate 签名支持 resource_type
        res = manager.allocate(req.worker_id, req.timeout, resource_type=req.type)
        return res
    except Exception as e:
        logger.error(f"Allocation failed: {e}")
        if "No resources available" in str(e) or "Pool for type" in str(e):
             raise HTTPException(status_code=503, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/release")
def release_resource(req: ReleaseReq, background_tasks: BackgroundTasks):
    background_tasks.add_task(manager.release, req.resource_id, req.worker_id)
    return {"status": "releasing"}

# [修改] 直接透传 None 给 Manager，由底层决定最终数值
@app.post("/query_rag")
def query_rag_service(req: RAGQueryReq):
    try:
        # 直接透传 None 给 Manager，由底层决定最终数值
        result_text = manager.query_rag(req.resource_id, req.worker_id, req.query, req.top_k)
        return {"status": "success", "results": result_text}
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.error(f"RAG Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
def get_status():
    return manager.get_status()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)