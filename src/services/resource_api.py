# src/services/resource_api.py
import sys
import os
import json
import re  # [新增]
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any, List  # 确保导入 List
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
        logger.error(f"Failed to parse JSON config after env substitution: {e}", exc_info=True)
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
        logger.critical(f"Critical startup error: {e}", exc_info=True)
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
            logger.error(f"Monitor error: {e}", exc_info=True)
        await asyncio.sleep(30)

# [修改] 更新 AllocReq 模型，增加 resource_types 字段
class AllocReq(BaseModel):
    worker_id: str
    timeout: float = 60.0
    type: str = "vm"  # 默认为 vm，兼容旧代码
    # [新增] 可选的资源类型列表
    resource_types: Optional[List[str]] = None

class ReleaseReq(BaseModel):
    resource_id: str
    worker_id: str

# [新增] 请求模型
class GetObsReq(BaseModel):
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
    # [新增] 检查 manager 是否已初始化
    if manager is None:
        logger.error("Resource Manager is not initialized.")
        raise HTTPException(status_code=503, detail="Service not initialized")

    # [Log] 记录分配请求的到达
    req_desc = req.resource_types if (req.resource_types and len(req.resource_types) > 0) else req.type
    logger.info(f"📥 [AllocReq] Worker={req.worker_id} requesting: {req_desc} (Timeout={req.timeout}s)")
    
    try:
        # 此时类型检查器知道 manager 一定不是 None，因为如果是 None 上面就抛异常了
        if req.resource_types and len(req.resource_types) > 0:
            result = manager.allocate_atomic(req.worker_id, req.resource_types, req.timeout)
        else:
            # [兼容] 走旧的单资源申请路径
            result = manager.allocate(req.worker_id, req.timeout, resource_type=req.type)
        
        # [Log] 记录分配成功
        logger.info(f"✅ [AllocOK] Worker={req.worker_id} acquired resources.")
        return result
            
    except Exception as e:
        # [Log] 记录分配失败，并包含完整堆栈信息
        logger.error(f"❌ [AllocFail] Worker={req.worker_id} failed: {e}", exc_info=True)
        if "No resources available" in str(e) or "timeout" in str(e).lower():
             raise HTTPException(status_code=503, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/release")
def release_resource(req: ReleaseReq, background_tasks: BackgroundTasks):
    # [新增] 检查 manager 是否已初始化
    if manager is None:
        logger.error("Resource Manager is not initialized.")
        raise HTTPException(status_code=503, detail="Service not initialized")
        
    # [Log] 记录释放请求
    logger.info(f"🗑️ [ReleaseReq] Worker={req.worker_id} releasing Resource={req.resource_id}")
    background_tasks.add_task(manager.release, req.resource_id, req.worker_id)
    return {"status": "releasing"}

# [修改] 直接透传 None 给 Manager，由底层决定最终数值
@app.post("/query_rag")
def query_rag_service(req: RAGQueryReq):
    # [新增] 检查 manager 是否已初始化
    if manager is None:
        logger.error("Resource Manager is not initialized.")
        raise HTTPException(status_code=503, detail="Service not initialized")
        
    try:
        # [Log] 记录RAG查询
        logger.info(f"🔍 [RAGQuery] Worker={req.worker_id} Resource={req.resource_id}")
        result_text = manager.query_rag(req.resource_id, req.worker_id, req.query, req.top_k)
        return {"status": "success", "results": result_text}
    except PermissionError as e:
        logger.warning(f"⚠️ [RAGQuery] Permission denied for {req.worker_id}: {e}")
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        # [Log] 记录RAG查询错误，并包含完整堆栈信息
        logger.error(f"❌ [RAGQuery] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
def get_status():
    # [新增] 检查 manager 是否已初始化
    if manager is None:
        logger.error("Resource Manager is not initialized.")
        raise HTTPException(status_code=503, detail="Service not initialized")
        
    return manager.get_status()

# [新增] 获取初始观测数据的 API
@app.post("/get_initial_observations")
def get_initial_observations_endpoint(req: GetObsReq):
    # [修改] 检查 manager 是否已初始化
    if manager is None:
        logger.error("Resource Manager is not initialized.")
        raise HTTPException(status_code=503, detail="Service not initialized")
        
    try:
        # Log
        logger.info(f"👁️ [GetObs] Worker={req.worker_id} requesting initial observations")
        
        # 调用 Manager 获取数据
        results = manager.get_initial_observations(req.worker_id)
        
        return {"status": "success", "observations": results}
    except Exception as e:
        logger.error(f"❌ [GetObs] Error: {e}", exc_info=True)
        # 失败时返回空字典，保证健壮性
        return {"status": "error", "message": str(e), "observations": {}}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)