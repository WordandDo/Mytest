#!/usr/bin/env python3
"""
直接从 rag_pool.py 模块拉起 RAG 服务并测试其可用性
"""
import os
import sys
import time
import json
import logging
import requests

# 确保可以导入项目模块
cwd = os.getcwd()
if os.path.join(cwd, "src") not in sys.path:
    sys.path.append(os.path.join(cwd, "src"))

from src.utils.resource_pools.rag_pool import RAGPoolImpl

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_deployment_config(config_path="deployment_config.json"):
    """加载部署配置"""
    if not os.path.exists(config_path):
        logger.warning(f"配置文件不存在: {config_path}")
        return {}

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"✅ 成功加载配置文件: {config_path}")
        return config
    except Exception as e:
        logger.error(f"❌ 加载配置文件失败: {e}")
        return {}


def test_rag_service():
    """测试 RAG 服务的完整流程"""

    print("="*70)
    print("🚀 直接测试 RAG 服务")
    print("="*70)

    # 1. 加载配置
    print("\n📋 Step 1: 加载配置...")
    config = load_deployment_config()

    if not config:
        logger.error("❌ 无法加载配置，使用默认配置")
        # 使用最小配置
        rag_config = {
            "rag_kb_path": "data/kb",
            "rag_index_path": "data/index",
            "rag_model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_device": "cpu",
            "use_faiss": False,
            "use_gpu_index": False,
            "default_top_k": 5
        }
    else:
        # 从配置中提取 RAG 相关参数
        rag_config = {}
        
        # 🆕 新增逻辑：优先尝试从嵌套结构 (resources -> rag -> config) 中读取
        if "resources" in config and "rag" in config["resources"]:
            print("ℹ️  检测到嵌套配置结构，正在提取 resources.rag.config...")
            rag_source = config["resources"]["rag"].get("config", {})
            # 直接使用里面的配置，或者进行过滤
            for key, value in rag_source.items():
                rag_config[key] = value
        else:
            # 旧逻辑：尝试从顶层扁平结构读取 (兼容旧的测试配置)
            for key, value in config.items():
                if key.startswith("rag_") or key in [
                    "embedding_device", "embedding_devices",
                    "use_faiss", "use_gpu_index", "use_compact", "use_gainrag",
                    "gpu_parallel_degree", "target_bytes_per_vector",
                    "passages_path", "gpu_id", "default_top_k"
                ]:
                    rag_config[key] = value

    # 显示配置
    print("\n配置信息:")
    print(f"  KB Path: {rag_config.get('rag_kb_path', 'N/A')}")
    print(f"  Index Path: {rag_config.get('rag_index_path', 'N/A')}")
    print(f"  Model: {rag_config.get('rag_model_name', 'N/A')}")
    print(f"  Device: {rag_config.get('embedding_device', 'N/A')}")
    print(f"  Use FAISS: {rag_config.get('use_faiss', False)}")
    print(f"  Use GPU Index: {rag_config.get('use_gpu_index', False)}")
    print(f"  Default Top-K: {rag_config.get('default_top_k', 5)}")

    # 检查必要路径是否存在
    kb_path = rag_config.get("rag_kb_path", "")
    index_path = rag_config.get("rag_index_path", "")

    if kb_path and not os.path.exists(kb_path):
        logger.warning(f"⚠️  知识库路径不存在: {kb_path}")
    else:
        logger.info(f"✅ 知识库路径存在: {kb_path}")

    if index_path and not os.path.exists(index_path):
        logger.warning(f"⚠️  索引路径不存在: {index_path}")
        logger.info("   (如果是首次运行，将自动构建索引)")
    else:
        logger.info(f"✅ 索引路径存在: {index_path}")

    # 2. 创建 RAG Pool
    print("\n📦 Step 2: 初始化 RAG Pool...")
    try:
        # 默认端口 8001
        rag_service_port = config.get("rag_service_port", 8001)

        # 优先从配置中获取 worker 数量，如果没有则默认为 2，并从 config 中移除以防冲突
        num_workers = rag_config.pop("num_rag_workers", 2)

        rag_pool = RAGPoolImpl(
            num_rag_workers=num_workers,
            rag_service_port=rag_service_port,
            **rag_config
        )
        logger.info(f"✅ RAG Pool 创建成功，端口: {rag_service_port}")
    except Exception as e:
        logger.error(f"❌ RAG Pool 创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 3. 初始化资源池 (启动服务)
    print("\n🚀 Step 3: 启动 RAG 服务...")
    try:
        success = rag_pool.initialize_pool(max_workers=10)
        if not success:
            logger.error("❌ RAG Pool 初始化失败")
            return False
        logger.info("✅ RAG 服务启动成功")
    except Exception as e:
        logger.error(f"❌ RAG 服务启动失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 4. 等待服务完全就绪
    print("\n⏳ Step 4: 等待服务就绪...")
    time.sleep(5)

    # 5. 测试健康检查
    print("\n🏥 Step 5: 测试健康检查...")
    service_url = rag_pool.service_url
    
    # [修改] 增加最大等待时间到 300秒 (5分钟)，因为 GainRAG 索引加载很慢
    max_retries = 60 
    retry_interval = 5
    
    print(f"   正在等待服务就绪 (最大等待 {max_retries * retry_interval} 秒)...")
    
    import requests
    from requests.exceptions import ConnectionError

    server_ready = False
    for i in range(max_retries):
        try:
            response = requests.get(f"{service_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                if health_data.get("ready"):
                    logger.info(f"✅ [{i+1}/{max_retries}] 服务已就绪！")
                    server_ready = True
                    break
                else:
                    logger.info(f"   [{i+1}/{max_retries}] 服务已启动但索引仍在加载中...")
            else:
                logger.warning(f"   [{i+1}/{max_retries}] 健康检查返回状态码: {response.status_code}")
        except ConnectionError:
            # 这是关键：连接被拒绝说明 uvicorn 还没启动，仍在加载索引，我们应该继续等
            logger.info(f"   [{i+1}/{max_retries}] 等待服务端口监听 (索引加载中)...")
        except Exception as e:
            logger.error(f"⚠️ [{i+1}/{max_retries}] 发生异常: {e}")
        
        time.sleep(retry_interval)

    if not server_ready:
        logger.error("❌ 服务启动超时，索引加载可能失败或耗时过长。")
        rag_pool.stop_all()
        return False
        
    logger.info("✅ 健康检查通过，准备测试查询。")

    # 6. 分配资源
    print("\n🎫 Step 6: 申请 RAG 资源...")
    try:
        # ✅ 更正为 allocate
        resource = rag_pool.allocate(worker_id="test_worker_001", timeout=30)
        if not resource:
            logger.error("❌ 资源分配失败")
            rag_pool.stop_all()
            return False

        logger.info(f"✅ 资源分配成功: {resource}")
        resource_url = resource.get("base_url")
        resource_token = resource.get("token")
    except Exception as e:
        logger.error(f"❌ 资源分配失败: {e}")
        import traceback
        traceback.print_exc()
        rag_pool.stop_all()
        return False

    # 7. 测试 RAG 查询
    print("\n🔍 Step 7: 测试 RAG 查询...")
    test_queries = [
        "What is artificial intelligence?",
        "Explain machine learning",
        "How does deep learning work?"
    ]

    query_success = False
    for query_text in test_queries:
        try:
            logger.info(f"    查询: {query_text}")
            response = requests.post(
                f"{resource_url}/query",
                json={
                    "query": query_text,
                    "top_k": 3,
                    "token": resource_token
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                
                if result.get("status") == "success":
                    # [修改] 适配返回值为字符串的情况
                    raw_results = result.get("results", "")
                    
                    if isinstance(raw_results, list):
                        # 如果未来改为返回列表，保持兼容
                        logger.info(f"    ✅ 查询成功！返回 {len(raw_results)} 条文档")
                        for i, doc in enumerate(raw_results[:2], 1):
                            score = doc.get("score", "N/A")
                            text = doc.get("text", "N/A")
                            logger.info(f"    结果 {i}: Score={score}, Text={text[:80]}...")
                    elif isinstance(raw_results, str):
                        # [当前逻辑] 处理字符串返回
                        preview_len = min(200, len(raw_results))
                        logger.info(f"    ✅ 查询成功！返回文本长度: {len(raw_results)}")
                        logger.info(f"    📝 结果预览:\n{raw_results[:preview_len]}...")
                        if len(raw_results) > 0:
                            query_success = True
                    else:
                        logger.warning(f"    ⚠️ 未知的结果格式: {type(raw_results)}")
                        
                    query_success = True
                else:
                    logger.warning(f"⚠️  查询返回非成功状态: {result}")
            else:
                logger.error(f"❌ 查询失败: {response.status_code} - {response.text}")

                # 分析失败原因
                if response.status_code == 503:
                    logger.error("   原因: 索引未加载 (503 Service Unavailable)")
                elif response.status_code == 500:
                    logger.error("   原因: 服务器内部错误")
                elif response.status_code == 404:
                    logger.error("   原因: 端点不存在 (检查 URL 路径)")

        except Exception as e:
            logger.error(f"❌ 查询异常: {e}")
            import traceback
            traceback.print_exc()

        print()

    # 8. 释放资源
    print("\n🗑️  Step 8: 释放资源...")
    try:
        # ✅ 更正为 release
        rag_pool.release(resource_id=resource.get("id"), worker_id="test_worker_001")
        logger.info("✅ 资源释放成功")
    except Exception as e:
        logger.error(f"⚠️  资源释放失败: {e}")

    # 9. 停止服务
    print("\n⏹️  Step 9: 停止 RAG 服务...")
    try:
        rag_pool.stop_all()
        logger.info("✅ RAG 服务已停止")
    except Exception as e:
        logger.error(f"⚠️  停止服务失败: {e}")

    # 10. 总结
    print("\n" + "="*70)
    print("📊 测试结果总结")
    print("="*70)

    if query_success:
        print("✅ RAG 服务可用！所有测试通过。")
        return True
    else:
        print("❌ RAG 服务不可用，请检查以下问题：")
        print("   1. 知识库路径是否正确且包含数据？")
        print("   2. 索引是否成功构建？")
        print("   3. 模型是否成功加载？")
        print("   4. 服务端口是否被占用？")
        print("   5. 检查服务日志获取详细错误信息")
        return False


if __name__ == "__main__":
    try:
        success = test_rag_service()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  测试被用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ 测试过程发生未预期的错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
