# RAG 多模式控制机制详解

## 问题回答

### 1. 如何实现只暴露一个工具或两个工具的控制？

通过 **Gateway 配置文件 + 工具注册表** 实现工具的选择性暴露。

### 2. 如何实现对不同 Gateway 文件的加载？

通过 **命令行参数传递 + 环境类加载机制** 实现配置文件的动态切换。

---

## 机制一：工具选择性暴露（Tool Registry + Gateway Config）

### 核心原理

系统使用 **装饰器注册 + 配置过滤** 的两阶段机制来控制工具暴露：

```
[阶段1: 工具注册]     [阶段2: 选择性暴露]
   装饰器               Gateway 配置
      ↓                      ↓
  工具注册表  --------→  MCP Server
(所有可用工具)        (暴露给 Agent 的工具)
```

### 实现细节

#### 步骤 1: 工具注册（装饰器）

**文件**: `src/mcp_server/rag_server.py` (行 130-175)

```python
from mcp_server.core.registry import ToolRegistry

# 注册密集检索工具到 "rag_query" 组
@ToolRegistry.register_tool("rag_query")
async def query_knowledge_base_dense(worker_id: str, query: str, top_k: Optional[int] = None) -> str:
    """
    [Dense Search] Query using semantic vector search (E5/Contriever).
    """
    return await _internal_query(worker_id, query, top_k, search_type="dense")

# 注册稀疏检索工具到 "rag_query_sparse" 组
@ToolRegistry.register_tool("rag_query_sparse")
async def query_knowledge_base_sparse(worker_id: str, query: str, top_k: Optional[int] = None) -> str:
    """
    [Sparse Search] Query using keyword matching (BM25).
    """
    return await _internal_query(worker_id, query, top_k, search_type="sparse")
```

**关键点**:
- `@ToolRegistry.register_tool(group_name)` 将函数注册到工具组
- 工具组名称是字符串，如 `"rag_query"` 或 `"rag_query_sparse"`
- 所有工具都被注册到全局 `ToolRegistry._REGISTRY` 字典中

#### 步骤 2: 工具注册表（Registry）

**文件**: `src/mcp_server/core/registry.py` (行 9-95)

```python
class ToolRegistry:
    """工具注册中心"""

    # 存储结构: {"group_name": [func1, func2, ...]}
    _REGISTRY: Dict[str, List[Callable]] = {}

    @classmethod
    def register_tool(cls, group_name: str, hidden: bool = False):
        """装饰器：将函数注册到指定工具组"""
        def decorator(func: Callable):
            if group_name not in cls._REGISTRY:
                cls._REGISTRY[group_name] = []

            if func not in cls._REGISTRY[group_name]:
                cls._REGISTRY[group_name].append(func)
                logger.debug(f"Registered tool '{func.__name__}' to group '{group_name}'")
            return func
        return decorator

    @classmethod
    def get_tools_by_group(cls, group_name: str) -> List[Callable]:
        """根据组名获取工具函数列表"""
        return cls._REGISTRY.get(group_name, [])

    @classmethod
    def get_tools_by_config(cls, module_config: dict) -> List[Callable]:
        """
        根据配置项解析需要的工具
        核心方法：从 gateway config 的 tool_groups 中提取工具
        """
        tools = []

        # 从配置中读取 tool_groups 列表
        groups = module_config.get("tool_groups", [])
        for group in groups:
            # 从注册表中提取对应组的工具
            tools.extend(cls.get_tools_by_group(group))

        # 去重
        seen = set()
        unique_tools = []
        for tool in tools:
            if tool not in seen:
                seen.add(tool)
                unique_tools.append(tool)

        return unique_tools
```

**关键点**:
- `_REGISTRY` 是全局字典，存储所有已注册的工具
- `get_tools_by_config()` 根据配置文件的 `tool_groups` 字段筛选工具
- 只有在 `tool_groups` 列表中的工具组才会被加载

#### 步骤 3: Gateway 配置文件（选择性暴露）

**三种配置示例**:

##### A. 混合模式（暴露两个工具）
**文件**: `gateway_config_rag_hybrid.json`

```json
{
  "modules": [
    {
      "resource_type": "rag",
      "tool_groups": [
        "rag_query",           ← 密集检索工具
        "rag_query_sparse"     ← 稀疏检索工具
      ]
    }
  ]
}
```

##### B. 仅密集检索（暴露一个工具）
**文件**: `gateway_config_rag_dense_only.json`

```json
{
  "modules": [
    {
      "resource_type": "rag",
      "tool_groups": [
        "rag_query"            ← 仅密集检索工具
      ]
    }
  ]
}
```

##### C. 仅稀疏检索（暴露一个工具）
**文件**: `gateway_config_rag_sparse_only.json`

```json
{
  "modules": [
    {
      "resource_type": "rag",
      "tool_groups": [
        "rag_query_sparse"     ← 仅稀疏检索工具
      ]
    }
  ]
}
```

#### 步骤 4: MCP Server 加载工具

**文件**: `src/mcp_server/main.py` (行 189-221)

```python
def main():
    # 1. 加载 Gateway 配置文件
    config = load_config(args.config)  # 读取 JSON 文件

    # 2. 自动发现所有工具（触发装饰器注册）
    ToolRegistry.autodiscover("mcp_server")

    # 3. 根据配置动态注册工具
    modules = config.get("modules", [])

    for module in modules:
        # 通过注册表获取该模块对应的工具函数
        tool_functions = ToolRegistry.get_tools_by_config(module)  # ← 核心调用

        for func in tool_functions:
            # 将函数注册为 MCP Tool（暴露给 Agent）
            mcp.tool()(func)
            logger.info(f"  + Registered tool: {func.__name__}")

    # 4. 启动 MCP Server
    mcp.run(transport='sse')
```

**执行流程**:

```
启动 MCP Server
    ↓
1. 读取 gateway_config.json
    ↓
2. 自动扫描并注册所有工具到 ToolRegistry
    ↓
3. 遍历配置中的 modules
    ↓
4. 根据 tool_groups 从 ToolRegistry 提取工具
    ↓
5. 将提取的工具注册到 MCP Server
    ↓
6. Agent 只能看到被暴露的工具
```

### 示例对比

| Gateway 配置 | tool_groups | Agent 可见工具 | 检索能力 |
|-------------|-------------|---------------|---------|
| hybrid | `["rag_query", "rag_query_sparse"]` | 2 个工具 | 密集 + 稀疏 |
| dense_only | `["rag_query"]` | 1 个工具 | 仅密集 |
| sparse_only | `["rag_query_sparse"]` | 1 个工具 | 仅稀疏 |

---

## 机制二：Gateway 配置文件的动态加载

### 核心原理

通过 **命令行参数 → 环境类 → MCP 客户端** 的传递链实现配置文件切换。

```
运行脚本
    ↓
--gateway_config_path 参数
    ↓
env_kwargs 字典
    ↓
HttpMCPEnv.__init__()
    ↓
_load_gateway_config()
    ↓
加载指定的 JSON 文件
```

### 实现细节

#### 步骤 1: 脚本传递参数

**文件**: `run_rag_env_multimode.sh` (行 108-113)

```bash
# 根据模式选择对应的 gateway 配置
case $MODE in
    hybrid)
        GATEWAY_CONFIG="gateway_config_rag_hybrid.json"
        ;;
    dense)
        GATEWAY_CONFIG="gateway_config_rag_dense_only.json"
        ;;
    sparse)
        GATEWAY_CONFIG="gateway_config_rag_sparse_only.json"
        ;;
esac

# 通过命令行参数传递给 Python 脚本
python src/run_parallel_rollout.py \
    --gateway_config_path "$GATEWAY_CONFIG" \  # ← 关键参数
    --data_path "$DATA_PATH" \
    --env_mode "http_mcp_rag"
```

#### 步骤 2: Python 脚本接收参数

**文件**: `src/run_parallel_rollout.py` (行 564 & 586)

```python
# 定义命令行参数
parser.add_argument(
    "--gateway_config_path",
    type=str,
    default="gateway_config.json",  # 默认配置
    help="Path to gateway config file"
)

args = parser.parse_args()

# 将参数传递给环境类
env_kwargs = {
    "mcp_server_url": args.mcp_server_url,
    "resource_api_url": args.resource_api_url,
    "gateway_config_path": args.gateway_config_path,  # ← 传递配置文件路径
}
```

#### 步骤 3: 环境类加载配置

**文件**: `src/envs/http_mcp_env.py` (行 66-67)

```python
class HttpMCPEnv:
    def __init__(self, model_name: str = "gpt-4", **kwargs):
        # 从 kwargs 中读取配置路径（默认值为 gateway_config.json）
        config_path = kwargs.get("gateway_config_path", "gateway_config.json")

        # 加载 Gateway 配置
        self.modules_config = self._load_gateway_config(config_path)
```

**文件**: `src/envs/http_mcp_env.py` (完整加载逻辑需要查看基类)

```python
def _load_gateway_config(self, config_path: str) -> Dict[str, Any]:
    """加载 Gateway 配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Gateway config not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    return config
```

#### 步骤 4: RAG 环境过滤配置

**文件**: `src/envs/http_mcp_rag_env.py` (行 85-110)

```python
class HttpMCPRagEnv(HttpMCPEnv):
    """RAG 专用环境，自动过滤非 RAG 模块"""

    def _load_gateway_config(self, config_path: str) -> Dict[str, Any]:
        """
        重写父类方法，仅加载 RAG 相关模块
        确保即使配置文件包含其他资源类型，也只使用 RAG 工具
        """
        # 先调用父类加载完整配置
        config = super()._load_gateway_config(config_path)

        # 过滤：仅保留 resource_type == "rag" 的模块
        if "modules" in config:
            config["modules"] = [
                module for module in config["modules"]
                if module.get("resource_type") == "rag"
            ]
            logger.info(f"Filtered config to {len(config['modules'])} RAG modules")

        return config
```

**关键点**:
- `HttpMCPRagEnv` 继承 `HttpMCPEnv`
- 重写 `_load_gateway_config()` 方法添加过滤逻辑
- 确保只加载 `resource_type == "rag"` 的模块
- 即使配置文件中包含 `system`、`vm_pyautogui` 等其他模块，也会被过滤掉

### 完整执行流程

```
1. 用户执行命令
   ./run_rag_env_multimode.sh dense

2. Shell 脚本设置变量
   GATEWAY_CONFIG="gateway_config_rag_dense_only.json"

3. 调用 Python 脚本
   python src/run_parallel_rollout.py \
       --gateway_config_path gateway_config_rag_dense_only.json

4. Python 解析命令行参数
   args.gateway_config_path = "gateway_config_rag_dense_only.json"

5. 创建环境实例
   env_kwargs = {"gateway_config_path": "gateway_config_rag_dense_only.json"}
   env = HttpMCPRagEnv(**env_kwargs)

6. 环境类加载配置
   config = _load_gateway_config("gateway_config_rag_dense_only.json")

7. 过滤 RAG 模块
   config["modules"] = [m for m in modules if m["resource_type"] == "rag"]

8. 结果：只加载 rag_query (密集检索) 工具
   Agent 只能看到一个检索工具
```

---

## 技术架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    用户层 (User Layer)                       │
│  ./run_rag_env_multimode.sh [hybrid|dense|sparse]          │
└──────────────────────────┬──────────────────────────────────┘
                           │ 选择配置文件
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                配置层 (Configuration Layer)                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────┐│
│  │  rag_hybrid.json │  │ rag_dense.json   │  │rag_sparse  ││
│  │ ["rag_query",    │  │ ["rag_query"]    │  │["rag_query ││
│  │  "rag_query_     │  │                  │  │ _sparse"]  ││
│  │  sparse"]        │  │                  │  │            ││
│  └──────────────────┘  └──────────────────┘  └────────────┘│
└──────────────────────────┬──────────────────────────────────┘
                           │ 传递给
                           ↓
┌─────────────────────────────────────────────────────────────┐
│              Python 脚本层 (Script Layer)                    │
│  run_parallel_rollout.py                                    │
│    ↓ --gateway_config_path                                  │
│  env_kwargs = {"gateway_config_path": "..."}                │
└──────────────────────────┬──────────────────────────────────┘
                           │ 初始化环境
                           ↓
┌─────────────────────────────────────────────────────────────┐
│              环境层 (Environment Layer)                      │
│  HttpMCPRagEnv.__init__()                                   │
│    ↓ _load_gateway_config()                                 │
│  读取 JSON → 过滤 RAG 模块 → 存储配置                        │
└──────────────────────────┬──────────────────────────────────┘
                           │ 配置决定工具
                           ↓
┌─────────────────────────────────────────────────────────────┐
│              MCP 服务器层 (MCP Server Layer)                 │
│  ToolRegistry                                               │
│  ┌──────────────┐        ┌──────────────┐                  │
│  │ rag_query    │        │ rag_query_   │                  │
│  │ (dense)      │        │ sparse       │                  │
│  └──────────────┘        └──────────────┘                  │
│         ↓                        ↓                          │
│  根据 tool_groups 选择性注册到 MCP Server                    │
└──────────────────────────┬──────────────────────────────────┘
                           │ 暴露工具
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                Agent 层 (Agent Layer)                        │
│  Agent 看到的工具:                                           │
│  • hybrid: 2 个工具 (dense + sparse)                        │
│  • dense:  1 个工具 (仅 dense)                              │
│  • sparse: 1 个工具 (仅 sparse)                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 关键代码路径总结

### 工具暴露控制

| 步骤 | 文件 | 行数 | 功能 |
|-----|------|------|------|
| 1 | `src/mcp_server/rag_server.py` | 130-175 | 使用装饰器注册工具到组 |
| 2 | `src/mcp_server/core/registry.py` | 9-95 | 工具注册表实现 |
| 3 | `gateway_config_rag_*.json` | - | 配置哪些工具组被暴露 |
| 4 | `src/mcp_server/main.py` | 189-221 | 根据配置加载工具 |

### 配置文件加载

| 步骤 | 文件 | 行数 | 功能 |
|-----|------|------|------|
| 1 | `run_rag_env_multimode.sh` | 108-113 | 脚本选择配置文件 |
| 2 | `src/run_parallel_rollout.py` | 564, 586 | 接收并传递参数 |
| 3 | `src/envs/http_mcp_env.py` | 66-67 | 读取配置路径 |
| 4 | `src/envs/http_mcp_rag_env.py` | 85-110 | 过滤 RAG 模块 |

---

## 设计优势

### 1. 解耦性
- 工具注册（代码）与工具暴露（配置）分离
- 修改暴露的工具只需改配置文件，无需改代码

### 2. 扩展性
- 添加新工具只需添加装饰器和配置项
- 支持任意组合的工具暴露方案

### 3. 灵活性
- 同一套代码支持多种运行模式
- 通过配置文件轻松切换模式

### 4. 可维护性
- 配置文件清晰表达工具暴露策略
- 代码逻辑简单，易于理解和调试

---

## 常见问题

### Q1: 如何添加新的工具？

1. 在 `rag_server.py` 中定义函数
2. 使用 `@ToolRegistry.register_tool("新组名")` 装饰
3. 在 gateway 配置的 `tool_groups` 中添加新组名

### Q2: 工具注册顺序重要吗？

不重要。`autodiscover()` 会扫描所有模块触发装饰器，顺序不影响结果。

### Q3: 可以在运行时切换配置吗？

不可以。配置在环境初始化时加载，运行时不可更改。需要重启任务并传入新配置。

### Q4: 如果配置文件路径错误会怎样？

环境类会抛出 `FileNotFoundError`，任务启动失败。

### Q5: 为什么需要过滤 `resource_type == "rag"`？

确保 RAG 环境只加载 RAG 工具，即使配置文件包含其他资源（如 `system`、`vm_pyautogui`）也会被忽略。

---

## 实验验证

你可以通过以下方式验证机制：

```bash
# 1. 运行混合模式，Agent 应该看到两个工具
./run_rag_env_multimode.sh hybrid

# 查看日志中的 "Registered tool:" 行，应该看到：
#   + Registered tool: query_knowledge_base_dense
#   + Registered tool: query_knowledge_base_sparse

# 2. 运行密集模式，Agent 应该只看到一个工具
./run_rag_env_multimode.sh dense

# 查看日志，应该只看到：
#   + Registered tool: query_knowledge_base_dense

# 3. 检查工具调用统计
# 查看 tool_stats/ 目录下的统计报告，验证 Agent 实际调用的工具
```

---

## 总结

1. **工具暴露控制** = 装饰器注册 + 配置文件过滤
2. **配置文件加载** = 命令行参数 → 环境类 → JSON 解析
3. **核心类**: `ToolRegistry` (工具管理), `HttpMCPRagEnv` (配置加载)
4. **核心方法**: `get_tools_by_config()` (工具筛选), `_load_gateway_config()` (配置加载)

这种设计实现了**配置驱动**的工具暴露策略，无需修改代码即可灵活控制 Agent 的能力边界。
