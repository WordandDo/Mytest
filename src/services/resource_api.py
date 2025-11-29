# src/services/resource_api.py
import sys
import os
import asyncio
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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ResourceAPI")

app = FastAPI()
manager: Optional[SimplifiedResourceManager] = None

def load_config() -> Dict[str, Any]:
    return {
        # VM 配置
        "provider_name": os.environ.get("PROVIDER_NAME", "aliyun"),
        "num_vms": int(os.environ.get("NUM_VMS", 2)),
        "region": os.environ.get("ALIYUN_REGION", "cn-hangzhou"),
        "snapshot_name": os.environ.get("SNAPSHOT_NAME", "init_state"),
        "os_type": "Ubuntu",
        "action_space": "computer_13",
        "screen_size": (1920, 1080),
        "headless": True,
        # RAG 配置 [新增]
        "num_rag_workers": int(os.environ.get("NUM_RAG_WORKERS", 2)),
        "rag_index_path": os.environ.get("RAG_INDEX_PATH", "src/data/rag_demo.jsonl")
    }

async def monitor_resource_usage():
    logger.info("Starting resource usage monitor (interval=30s)...")
    while True:
        try:
            if manager:
                stats = manager.get_status()
                # [修改] 打印 VM 和 RAG 的状态
                log_msg = "📊 [Monitor] "
                if "vm" in stats:
                    s = stats["vm"]
                    log_msg += f"VM(Free:{s.get('free')}/{s.get('total')}) "
                if "rag" in stats:
                    s = stats["rag"]
                    log_msg += f"RAG(Free:{s.get('free')}/{s.get('total')})"
                logger.info(log_msg)
        except Exception as e:
            logger.error(f"Monitor error: {e}")
        await asyncio.sleep(30)

@app.on_event("startup")
async def startup_event():
    global manager
    config = load_config()
    manager = SimplifiedResourceManager(config)
    
    loop = asyncio.get_running_loop()
    success = await loop.run_in_executor(None, manager.initialize)
    
    if not success:
        logger.error("Failed to start Resource Manager (some pools may be offline)!")
    
    asyncio.create_task(monitor_resource_usage())

# [修改] 请求模型增加 resource_type
class AllocReq(BaseModel):
    worker_id: str
    timeout: float = 60.0
    type: str = "vm"  # 默认为 vm，兼容旧代码

class ReleaseReq(BaseModel):
    resource_id: str
    worker_id: str

@app.post("/allocate")
def allocate_resource(req: AllocReq):
    try:
        # 传递 type 参数
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

@app.get("/status")
def get_status():
    return manager.get_status()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)