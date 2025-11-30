# src/utils/resource_pools/factory.py
import importlib
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

class ResourcePoolFactory:
    @staticmethod
    def create_pool(class_path: str, config: Dict[str, Any]) -> Any:
        """
        动态加载并实例化资源池。
        :param class_path: 类路径，例如 "utils.resource_pools.vm_pool.VMPoolImpl"
        :param config: 包含构造函数参数的字典
        """
        try:
            # 1. 解析模块名和类名
            module_name, class_name = class_path.rsplit('.', 1)
            
            # 2. 动态导入模块
            module = importlib.import_module(module_name)
            
            # 3. 获取类
            cls = getattr(module, class_name)
            
            # 4. 实例化 (将 config 字典解包传入构造函数)
            # 注意：VMPoolImpl(num_vms=...) 和 RAGPoolImpl(num_rag_workers=...) 
            # 的参数名必须在 config 字典中存在
            instance = cls(**config)
            
            return instance
            
        except ImportError as e:
            logger.error(f"Failed to import module for {class_path}: {e}")
            raise
        except AttributeError as e:
            logger.error(f"Class {class_name} not found in module {module_name}: {e}")
            raise
        except TypeError as e:
            logger.error(f"Config mismatch for {class_name}. Check json keys vs __init__ args: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error creating pool {class_path}: {e}")
            raise