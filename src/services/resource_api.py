# src/services/resource_api.py
# 这是一个基于FastAPI的资源管理服务，负责提供RESTful API接口来管理各种资源（如虚拟机、RAG等）
# 采用"通用资源调度框架 + 特定业务逻辑插件"的设计模式，实现资源的动态管理和扩展

import sys
import os
import json
import re  # 用于环境变量替换的正则表达式处理
import asyncio
from dotenv import load_dotenv  # 用于加载.env文件中的环境变量
from fastapi import FastAPI, HTTPException, BackgroundTasks  # FastAPI框架相关组件
from pydantic import BaseModel  # 用于定义请求和响应的数据模型
from typing import Optional, Dict, Any, List  # 类型注解支持
import logging  # 日志记录
import uvicorn  # ASGI服务器，用于运行FastAPI应用

# 将当前工作目录和src目录添加到Python路径中，确保可以正确导入自定义模块
cwd = os.getcwd()
sys.path.append(cwd)
sys.path.append(os.path.join(cwd, "src"))

# 导入资源管理器类，负责具体的资源管理逻辑
from services.simple_manager import GenericResourceManager

# 1. 加载 .env 到环境变量，使配置中的${VAR_NAME}形式的环境变量能够被正确替换
load_dotenv()

# 配置日志格式和级别
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ResourceAPI")

# 创建FastAPI应用实例
app = FastAPI()

# 全局资源管理器实例，初始为None，在应用启动时初始化
# 使用Optional类型注解表明该变量可能为None
manager: Optional[GenericResourceManager] = None

# =========================================================================
# [基础设施配置部分]
# 负责加载配置、启动事件、资源监控等基础功能
# =========================================================================

# [新增] 带有环境变量替换功能的配置加载器
# 支持在配置文件中使用${VAR_NAME}的形式引用环境变量
def load_deployment_config(path: str = "deployment_config.json") -> Dict[str, Any]:
    """
    加载部署配置文件，并处理其中的环境变量替换
    
    Args:
        path: 配置文件路径，默认为"deployment_config.json"
        
    Returns:
        解析后的配置字典
        
    Raises:
        FileNotFoundError: 配置文件不存在
        json.JSONDecodeError: 配置文件格式错误
    """
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
            # 环境变量未找到时发出警告并使用空字符串，避免程序崩溃
            logger.warning(f"⚠️ Environment variable {var_name} not found in .env")
            return "" 
        return value

    # 执行替换，将配置中的${VAR_NAME}替换为实际的环境变量值
    content_with_env = re.sub(r'\$\{(\w+)\}', replace_env, content)
    
    try:
        return json.loads(content_with_env)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON config after env substitution: {e}", exc_info=True)
        raise

@app.on_event("startup")
async def startup_event():
    """
    应用启动事件处理函数
    负责初始化资源管理器并启动资源监控任务
    """
    global manager
    try:
        # 2. 加载统一配置
        config = load_deployment_config("deployment_config.json")
        
        # 3. 初始化通用管理器
        manager = GenericResourceManager(config)
        
        # 在executor中运行初始化过程，避免阻塞事件循环
        loop = asyncio.get_running_loop()
        success = await loop.run_in_executor(None, manager.initialize)
        
        if not success:
            logger.error("Failed to start Resource Manager (some pools may be offline)!")
        
        # 启动资源使用监控任务
        asyncio.create_task(monitor_resource_usage())
    except Exception as e:
        logger.critical(f"Critical startup error: {e}", exc_info=True)
        sys.exit(1)

async def monitor_resource_usage():
    """
    资源使用监控任务
    定期输出各资源池的状态信息，便于监控系统运行状况
    """
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

# =========================================================================
# [标准请求模型部分]
# 定义各种API接口的请求数据模型
# =========================================================================

# [修改] 更新 AllocReq 模型
class AllocReq(BaseModel):
    """
    资源分配请求模型
    
    Attributes:
        worker_id: 工作节点ID
        timeout: 分配超时时间（秒）
        type: 资源类型（可选，用于单资源分配）
        resource_types: 资源类型列表（可选，用于批量资源分配）
    """
    worker_id: str
    timeout: float = 60.0
    
    # [关键修改] 移除 "vm" 默认值，设为 Optional
    # 这样高层如果不传 type，就不会默认指向 "vm"，而是由 resource_types 决定
    type: Optional[str] = None 
    
    # 推荐使用列表方式申请
    resource_types: Optional[List[str]] = None

class ReleaseReq(BaseModel):
    """
    资源释放请求模型
    
    Attributes:
        resource_id: 资源ID
        worker_id: 工作节点ID
    """
    resource_id: str
    worker_id: str

# [新增] 请求模型
class GetObsReq(BaseModel):
    """
    获取初始观测数据请求模型
    
    Attributes:
        worker_id: 工作节点ID
    """
    worker_id: str

# [修改] 将top_k改为Optional，默认为None，表示"使用服务器配置的默认值"
class RAGQueryReq(BaseModel):
    """
    RAG查询请求模型
    
    Attributes:
        resource_id: 资源ID
        worker_id: 工作节点ID
        query: 查询内容
        top_k: 返回结果数量（可选，None表示使用服务器默认值）
    """
    resource_id: str
    worker_id: str
    query: str
    # [修改] 改为 Optional，默认为 None，表示"使用服务器配置的默认值"
    top_k: Optional[int] = None

# =========================================================================
# [标准资源生命周期接口]
# 提供适用于所有资源类型的通用操作接口
# =========================================================================

@app.post("/allocate")
def allocate_resource(req: AllocReq):
    """
    分配资源接口
    
    支持两种分配方式：
    1. 单资源分配：通过type字段指定资源类型
    2. 批量资源分配：通过resource_types字段指定资源类型列表
    
    Args:
        req: 资源分配请求
        
    Returns:
        分配成功的资源信息
        
    Raises:
        HTTPException: 资源分配失败或超时
    """
    # [新增] 检查 manager 是否已初始化
    if manager is None:
        logger.error("Resource Manager is not initialized.")
        raise HTTPException(status_code=503, detail="Service not initialized")

    # [新增] 校验：必须至少指定一种资源
    if not req.resource_types and not req.type:
        raise HTTPException(status_code=400, detail="Must specify 'resource_types' (list) or 'type' (string)")

    # [Log] 记录分配请求的到达
    req_desc = req.resource_types if (req.resource_types and len(req.resource_types) > 0) else req.type
    logger.info(f"📥 [AllocReq] Worker={req.worker_id} requesting: {req_desc} (Timeout={req.timeout}s)")
    
    try:
        # 此时类型检查器知道 manager 一定不是 None，因为如果是 None 上面就抛异常了
        if req.resource_types and len(req.resource_types) > 0:
            # 批量资源分配
            result = manager.allocate_atomic(req.worker_id, req.resource_types, req.timeout)
        else:
            # 单资源申请，明确传入 req.type
            result = manager.allocate(req.worker_id, req.timeout, resource_type=req.type)
        
        # [Log] 记录分配成功
        logger.info(f"✅ [AllocOK] Worker={req.worker_id} acquired resources.")
        return result
            
    except Exception as e:
        # [Log] 记录分配失败，并包含完整堆栈信息
        logger.error(f"❌ [AllocFail] Worker={req.worker_id} failed: {e}", exc_info=True)
        if "No resources available" in str(e) or "timeout" in str(e).lower():
             raise HTTPException(status_code=503, detail=str(e))
        # 如果是资源未找到 (e.g. key mismatch)，也会在这里捕获
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/release")
def release_resource(req: ReleaseReq, background_tasks: BackgroundTasks):
    """
    释放资源接口
    
    采用后台任务方式执行资源释放，提高接口响应速度
    
    Args:
        req: 资源释放请求
        background_tasks: FastAPI后台任务管理器
        
    Returns:
        释放状态信息
    """
    # [新增] 检查 manager 是否已初始化
    if manager is None:
        logger.error("Resource Manager is not initialized.")
        raise HTTPException(status_code=503, detail="Service not initialized")
        
    # [Log] 记录释放请求
    logger.info(f"🗑️ [ReleaseReq] Worker={req.worker_id} releasing Resource={req.resource_id}")
    # 将资源释放操作添加到后台任务队列中执行
    background_tasks.add_task(manager.release, req.resource_id, req.worker_id)
    return {"status": "releasing"}

@app.get("/status")
def get_status():
    """
    获取资源状态接口
    
    返回所有已初始化资源池的当前状态信息
    
    Returns:
        各资源池的状态信息字典
    """
    # [新增] 检查 manager 是否已初始化
    if manager is None:
        logger.error("Resource Manager is not initialized.")
        raise HTTPException(status_code=503, detail="Service not initialized")
        
    return manager.get_status()

# =========================================================================
# [特定资源操作接口]
# 为特定资源类型提供的专用操作接口
# =========================================================================

# [修改] 直接透传 None 给 Manager，由底层决定最终数值
@app.post("/query_rag")
def query_rag_service(req: RAGQueryReq):
    """
    RAG查询接口
    
    Args:
        req: RAG查询请求
        
    Returns:
        查询结果
        
    Raises:
        HTTPException: 查询过程中出现错误
    """
    # [新增] 检查 manager 是否已初始化
    if manager is None:
        logger.error("Resource Manager is not initialized.")
        raise HTTPException(status_code=503, detail="Service not initialized")
        
    try:
        # [Log] 记录RAG查询
        logger.info(f"🔍 [RAGQuery] Worker={req.worker_id} Resource={req.resource_id}")
        # 调用管理器执行RAG查询
        result_text = manager.query_rag(req.resource_id, req.worker_id, req.query, req.top_k)
        return {"status": "success", "results": result_text}
    except PermissionError as e:
        logger.warning(f"⚠️ [RAGQuery] Permission denied for {req.worker_id}: {e}")
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        # [Log] 记录RAG查询错误，并包含完整堆栈信息
        logger.error(f"❌ [RAGQuery] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# [新增] 获取初始观测数据的 API
@app.post("/get_initial_observations")
def get_initial_observations_endpoint(req: GetObsReq):
    """
    获取初始观测数据接口
    
    用于获取工作节点下所有资源的初始观测数据，如虚拟机的屏幕截图等
    
    Args:
        req: 获取观测数据请求
        
    Returns:
        观测数据字典
    """
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

# 应用入口点
if __name__ == "__main__":
    # 使用uvicorn运行FastAPI应用
    uvicorn.run(app, host="0.0.0.0", port=8000)