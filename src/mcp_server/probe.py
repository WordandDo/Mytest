import httpx
import asyncio
import logging

logger = logging.getLogger("ResourceProbe")

async def wait_for_resource_availability(
    api_url: str, 
    resource_type: str, 
    timeout: int = 30, 
    interval: float = 2.0
) -> bool:
    """
    资源探活探针：轮询 Resource API 的 /status 接口，直到有空闲资源或超时。
    
    :param api_url: Resource API 地址 (e.g. http://localhost:8000)
    :param resource_type: "vm" 或 "rag"
    :param timeout: 最大等待时间（秒）
    :param interval: 轮询间隔（秒）
    :return: True (有资源) / False (超时无资源)
    """
    start_time = asyncio.get_event_loop().time()
    
    logger.info(f"🔍 Probing for {resource_type} resources at {api_url}...")
    
    async with httpx.AsyncClient() as client:
        while (asyncio.get_event_loop().time() - start_time) < timeout:
            try:
                # 1. 获取状态
                resp = await client.get(f"{api_url}/status", timeout=5.0)
                resp.raise_for_status()
                status_data = resp.json()
                
                # 2. 解析特定资源池
                pool_stats = status_data.get(resource_type, {})
                free_count = pool_stats.get("free", 0)
                total_count = pool_stats.get("total", 0)
                
                # 3. 判断是否有空闲
                if free_count > 0:
                    logger.info(f"✅ Resource {resource_type} available (Free: {free_count}/{total_count})")
                    return True
                else:
                    logger.debug(f"⏳ Waiting for {resource_type}... (Free: 0/{total_count})")
            
            except Exception as e:
                logger.warning(f"⚠️ Probe failed: {e}")
            
            # 等待下一次检查
            await asyncio.sleep(interval)
            
    logger.error(f"❌ Probe timeout: No {resource_type} resources available after {timeout}s")
    return False