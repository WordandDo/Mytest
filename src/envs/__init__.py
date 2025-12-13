"""
Environment package for AgentFlow.
"""

# Import data models from data_models module (no dependencies)
from .data_models import (
    Observation,
    TrajectoryStep,
    TaskTrajectory
)

# Lazy imports for concrete environments to avoid tool dependency issues at import time
def __getattr__(name):
    """Lazy import for environment classes."""
    
    # 1. 基础环境
    if name == "Environment":
        # 兼容旧引用，直接映射到当前唯一的 MCP 环境基类
        from .http_mcp_env import HttpMCPEnv as Environment
        return Environment
    elif name == "HttpMCPEnv":
        from .http_mcp_env import HttpMCPEnv
        return HttpMCPEnv
    elif name == "Tool":
        from tools.tool import Tool
        return Tool
        
    # 2. RAG 环境 (修正映射关系)
    elif name == "RAGEnvironment":
        # [关键修正] 将 RAGEnvironment 映射到 http_mcp_rag_env 模块
        from .http_mcp_rag_env import HttpMCPRagEnv as RAGEnvironment
        return RAGEnvironment
    elif name == "create_rag_environment":
        # 简单的工厂函数封装
        def create_rag_environment(**kwargs):
            from .http_mcp_rag_env import HttpMCPRagEnv
            return HttpMCPRagEnv(**kwargs)
        return create_rag_environment
    elif name == "HttpMCPVmEnv":
        from .http_mcp_vm_env import HttpMCPVmEnv
        return HttpMCPVmEnv

    # 3. 其他环境 (Math, Python, Web, OSWorld)
    # 注意：确保这些对应的 .py 文件真实存在，否则也会报类似的错
    elif name == "MathEnvironment":
        try:
            from .math_environment import MathEnvironment
            return MathEnvironment
        except ImportError:
            # 容错处理：如果文件不存在，抛出更清晰的错误
            raise ImportError("MathEnvironment module is missing (src/envs/math_environment.py)")
            
    elif name == "PythonEnvironment":
        try:
            from .python_environment import PythonEnvironment
            return PythonEnvironment
        except ImportError:
            raise ImportError("PythonEnvironment module is missing (src/envs/python_environment.py)")
            
    elif name == "WebEnvironment":
        try:
            from .web_environment import WebEnvironment
            return WebEnvironment
        except ImportError:
            raise ImportError("WebEnvironment module is missing (src/envs/web_environment.py)")
            
    elif name == "TBenchEnvironment":
        try:
            from .tbench_environment import TBenchEnvironment
            return TBenchEnvironment
        except ImportError:
            raise ImportError("TBenchEnvironment module is missing (src/envs/tbench_environment.py)")
            
    elif name == "OSWorldEnvironment":
        try:
            from .osworld_environment import OSWorldEnvironment
            return OSWorldEnvironment
        except ImportError:
            raise ImportError("OSWorldEnvironment module is missing (src/envs/osworld_environment.py)")

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    # Data models
    "Observation",
    "TrajectoryStep",
    "TaskTrajectory",
    # Base classes (Environment is kept as alias to HttpMCPEnv for compatibility)
    "Environment",
    "HttpMCPEnv",
    "Tool",
    # Environments (lazy loaded)
    "MathEnvironment",
    "PythonEnvironment",
    "RAGEnvironment",  # <--- 现在这个名字是安全的别名
    "WebEnvironment",
    "TBenchEnvironment",
    "OSWorldEnvironment",
    "HttpMCPVmEnv",
    # Factory functions
    "create_rag_environment"
]
