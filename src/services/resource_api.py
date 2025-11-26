# src/services/resource_api.py
import sys
import os
# [新增] 引入 dotenv
from dotenv import load_dotenv 

# 1. 加载 .env 文件
# 这行代码会自动在当前目录及父级目录寻找 .env 文件，并将其内容注入到 os.environ 中
load_dotenv()

# [可选] 调试打印，确认是否加载成功 (切勿在生产环境打印完整 Key)
if os.environ.get("ALIYUN_ACCESS_KEY_ID"):
    print("✅ Aliyun credentials found in environment.")
else:
    print("⚠️ Warning: ALIYUN_ACCESS_KEY_ID not found.")

# 2. 处理导入路径 (您之前的代码)
cwd = os.getcwd()
sys.path.append(cwd)
sys.path.append(os.path.join(cwd, "src"))
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
import uvicorn

# 引入刚才写的简化管理器
from services.simple_manager import SimplifiedResourceManager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ResourceAPI")

app = FastAPI()

# 全局单例
manager: Optional[SimplifiedResourceManager] = None
def load_config() -> Dict[str, Any]:
    """
    加载配置。
    遵循 VMPoolImpl 的原生参数标准，保持参数扁平化。
    """
    return {
        # 1. 核心 Provider 配置
        "provider_name": "aliyun",
        "num_vms": 2,  # 您希望启动的实例数量
        
        # 2. 区域 (优先读取环境变量，保持与 .env 一致)
        "region": os.environ.get("ALIYUN_REGION", "cn-hangzhou"),
        
        # 3. VMPoolImpl 标准参数 (保持默认或根据需求微调)
        # 注意：这里不传入 extra_kwargs 字典，也不传入 instance_type
        # 让 Provider 自动使用 .env 中的配置或默认规格
        "snapshot_name": "init_state",
        "os_type": "Ubuntu",
        "action_space": "computer_13",
        "screen_size": (1920, 1080),
        "headless": True, # 云服务器通常为 True
    }

@app.on_event("startup")
def startup_event():
    global manager
    config = load_config()
    manager = SimplifiedResourceManager(config)
    if not manager.initialize():
        logger.error("Failed to start Resource Manager!")
        # 在生产环境中这里可能需要 sys.exit(1)

@app.on_event("shutdown")
def shutdown_event():
    # 如果需要清理资源，可以在这里加 manager.stop_all()
    pass

# --- API 定义 ---

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
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/release")
def release_resource(req: ReleaseReq, background_tasks: BackgroundTasks):
    # 使用 BackgroundTasks 异步释放，让 HTTP 接口立即返回
    # 这样 Client 不用等 VM 重置完成，提高并发响应速度
    background_tasks.add_task(manager.release, req.resource_id, req.worker_id)
    return {"status": "releasing"}

@app.get("/status")
def get_status():
    return manager.get_status()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)