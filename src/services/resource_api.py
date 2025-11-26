# src/services/resource_api.py
import sys
import os
import asyncio  # [新增]
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
import uvicorn

cwd = os.getcwd()
sys.path.append(cwd)
sys.path.append(os.path.join(cwd, "src"))

from services.simple_manager import SimplifiedResourceManager

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s' # [优化日志格式]
)
logger = logging.getLogger("ResourceAPI")

app = FastAPI()
manager: Optional[SimplifiedResourceManager] = None

# ... (load_config 函数保持不变) ...
def load_config() -> Dict[str, Any]:
    # ... (保持原样)
    return {
        "provider_name": os.environ.get("PROVIDER_NAME", "aliyun"),
        "num_vms": int(os.environ.get("NUM_VMS", 2)),
        "region": os.environ.get("ALIYUN_REGION", "cn-hangzhou"),
        "snapshot_name": os.environ.get("SNAPSHOT_NAME", "init_state"),
        "os_type": "Ubuntu",
        "action_space": "computer_13",
        "screen_size": (1920, 1080),
        "headless": True,
    }

# [新增] 后台监控协程
async def monitor_resource_usage():
    logger.info("Starting resource usage monitor (interval=30s)...")
    while True:
        try:
            if manager and manager.pool:
                stats = manager.get_status()
                # 打印清晰的统计条
                logger.info(
                    f"📊 [Monitor] Total: {stats.get('total', 0)} | "
                    f"Free: {stats.get('free', 0)} | "
                    f"Occupied: {stats.get('occupied', 0)} | "
                    f"Error: {stats.get('error', 0)} | "
                    f"Allocations: {stats.get('allocations', 0)}"
                )
        except Exception as e:
            logger.error(f"Monitor error: {e}")
        await asyncio.sleep(30)

@app.on_event("startup")
async def startup_event(): # [修改] 改为 async
    global manager
    config = load_config()
    manager = SimplifiedResourceManager(config)
    
    # 启动时初始化（这里使用 run_in_executor 避免阻塞事件循环）
    loop = asyncio.get_running_loop()
    success = await loop.run_in_executor(None, manager.initialize)
    
    if not success:
        logger.error("Failed to start Resource Manager!")
        # sys.exit(1) 
    
    # [新增] 启动监控任务
    asyncio.create_task(monitor_resource_usage())

# ... (其余 API 接口 AllocReq, ReleaseReq, allocate_resource, release_resource, get_status 保持不变) ...
# 请确保保留原有的代码逻辑

class AllocReq(BaseModel):
    worker_id: str
    timeout: float = 60.0

class ReleaseReq(BaseModel):
    resource_id: str
    worker_id: str

@app.post("/allocate")
def allocate_resource(req: AllocReq):
    try:
        res = manager.allocate(req.worker_id, req.timeout)
        return res
    except Exception as e:
        logger.error(f"Allocation failed: {e}")
        if "No resources available" in str(e):
             raise HTTPException(status_code=503, detail="Resource pool exhausted")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/release")
def release_resource(req: ReleaseReq, background_tasks: BackgroundTasks):
    background_tasks.add_task(manager.release, req.resource_id, req.worker_id)
    return {"status": "releasing"}

@app.get("/status")
def get_status():
    return manager.get_status()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)