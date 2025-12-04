基于我们刚才对 `vm_computer_13` 和 `vm_pyautogui` 的解耦重构工作，以及 `system_tools.py` 的动态加载机制，我为您整理了一份**MCP 资源模块开发指南与命名规范**。

这份指南旨在确保新增加的资源模块能够被 Gateway 自动识别、正确初始化，并避免工具注册冲突。

-----

# MCP Gateway 资源模块开发指南

## 1\. 核心架构原理

MCP Gateway 采用**约定优于配置**的动态加载机制：

1.  **动态分发**：`system_tools.py` 根据请求中的 `resource_type` 字符串，动态拼接模块名和函数名来加载资源。
2.  **独立会话**：每个资源 Server 文件维护自己的 `GLOBAL_SESSIONS` 字典。
3.  **会话同步**：Gateway 负责将底层资源（如 IP、Port）写入对应模块的 `GLOBAL_SESSIONS` 中。

-----

## 2\. 严格命名规范 (Naming Conventions)

为了保证系统能自动找到您的代码，必须严格遵守以下命名规则。假设您的新资源类型名称为 **`my_new_resource`**。

### 2.1 文件与模块命名

| 对象 | 命名规则 | 示例 | 说明 |
| :--- | :--- | :--- | :--- |
| **文件名** | `src/mcp_server/{resource_type}_server.py` | `src/mcp_server/my_new_resource_server.py` | 必须位于 `mcp_server` 包下 |
| **Gateway配置** | `resource_type` | `"resource_type": "my_new_resource"` | 在 `gateway_config.json` 中配置 |

### 2.2 函数与变量命名 (Python 代码内)

| 对象 | 命名规则 | 示例 | 必须性 |
| :--- | :--- | :--- | :--- |
| **初始化入口函数** | `{resource_type}_initialization` | `async def my_new_resource_initialization(...)` | **必须** (Gateway 反射调用) |
| **全局会话变量** | `GLOBAL_SESSIONS` | `GLOBAL_SESSIONS = {}` | **必须** (Gateway 写入会话) |
| **Setup 工具** | `setup_{resource_type}_session` | `async def setup_my_new_resource_session(...)` | **必须** (避免注册冲突) |
| **Teardown 工具** | `teardown_{resource_type}_env` | `async def teardown_my_new_resource_env(...)` | **必须** (避免注册冲突) |

-----

## 3\. 开发步骤详解

### 步骤 1: 创建 Server 文件

在 `src/mcp_server/` 目录下新建文件，例如 `my_new_resource_server.py`。

### 步骤 2: 编写标准模版代码

复制以下模版，并将 `my_new_resource` 替换为您的实际资源名称。

```python
# src/mcp_server/my_new_resource_server.py
import json
import logging
import httpx
from mcp.server.fastmcp import FastMCP
from mcp_server.core.registry import ToolRegistry

# 1. 定义 Logger 和 Server
logger = logging.getLogger("MyNewResourceServer")
mcp = FastMCP("My New Resource Gateway")

# 2. 定义全局会话字典 (必须叫 GLOBAL_SESSIONS)
GLOBAL_SESSIONS = {}

# 3. 定义初始化入口 (必须匹配 {resource_type}_initialization)
# system_tools.py 会动态查找并调用此函数
async def my_new_resource_initialization(worker_id: str, config_content = None) -> bool:
    logger.info(f"[{worker_id}] Initializing my_new_resource...")
    try:
        # 检查会话是否存在
        if worker_id not in GLOBAL_SESSIONS:
            # 如果不存在，调用下方的 Setup 工具进行申请
            await setup_my_new_resource_session(
                config_name="auto_init", 
                task_id="unknown", 
                worker_id=worker_id, 
                init_script=json.dumps(config_content) if config_content else ""
            )
        
        # 执行具体的初始化逻辑 (如连接数据库、设置环境等)
        # ... 您的业务逻辑 ...
        
        return True
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        return False

# 4. 注册生命周期工具 (函数名必须包含资源前缀，避免冲突)
@ToolRegistry.register_tool("computer_lifecycle", hidden=True)
async def setup_my_new_resource_session(config_name: str, task_id: str, worker_id: str, init_script: str = "") -> str:
    """
    Allocates and sets up my_new_resource.
    """
    # 硬编码资源类型，向 Resource API 申请资源
    target_resource_type = "my_new_resource" 
    
    # ... (参考 vm_pyautogui_server.py 中的 HTTP 请求逻辑) ...
    # 成功后写入 GLOBAL_SESSIONS[worker_id]
    return json.dumps({"status": "success"})

@ToolRegistry.register_tool("computer_lifecycle", hidden=True)
async def teardown_my_new_resource_env(worker_id: str) -> str:
    # ... 清理逻辑 ...
    if worker_id in GLOBAL_SESSIONS:
        del GLOBAL_SESSIONS[worker_id]
    return "Released"

# 5. 注册业务工具
@ToolRegistry.register_tool("my_resource_actions")
async def my_custom_action(worker_id: str, command: str) -> str:
    """Execute a custom command."""
    session = GLOBAL_SESSIONS.get(worker_id)
    if not session:
        return "Error: No session found"
    # ... 执行逻辑 ...
    return "Executed"
```

### 步骤 3: 注册 VM 会话同步 (如果是 VM 类型)

如果您的新资源属于虚拟机 (VM) 类型（即包含 IP 和 Port，需要 `PythonController` 连接），您需要修改 `src/mcp_server/system_tools.py`。

找到 `_sync_resource_sessions` 函数，在 `vm_module_map` 中添加映射：

```python
# src/mcp_server/system_tools.py

async def _sync_resource_sessions(...):
    # ...
    vm_module_map = {
        "vm_pyautogui": "mcp_server.vm_pyautogui_server",
        "vm_computer_13": "mcp_server.vm_computer_13_server",
        # [新增] 注册您的新 VM 资源
        "my_new_resource": "mcp_server.my_new_resource_server" 
    }
    # ...
```

*注：如果不是 VM 类型（如 API 服务、数据库连接），则无需此步，只要在初始化函数里自行处理连接即可。*

### 步骤 4: 更新 Gateway 配置

在 `gateway_config.json` 或 `config.json` 的 `modules` 列表中注册该模块，以便 Gateway 启动时加载这些工具。

```json
{
  "modules": [
    {
      "resource_type": "my_new_resource",
      "tool_groups": ["computer_lifecycle", "my_resource_actions"]
    }
  ]
}
```

-----

## 4\. 常见问题排查 (Troubleshooting)

1.  **"No init logic defined"**:

      * **原因**: `system_tools.py` 找不到 `{resource_type}_initialization` 函数。
      * **检查**: 确保文件名正确，且函数名严格拼写为 `资源类型 + _initialization`。

2.  **"Tool already exists"**:

      * **原因**: 不同的 Server 文件中使用了相同的工具函数名（如都叫 `setup_vm_session`）。
      * **检查**: 确保 `@ToolRegistry.register_tool` 装饰的函数名是全局唯一的（加上前缀）。

3.  **Session not found / controller is None**:

      * **原因**: Gateway 将资源信息写错了地方，或者读取的地方不对。
      * **检查**:
          * 确认 Server 文件中定义了 `GLOBAL_SESSIONS = {}`。
          * 如果是 VM 资源，确认 `system_tools.py` 的映射表已更新。

4.  **AttributeError: module '...' has no attribute 'GLOBAL\_SESSIONS'**:

      * **原因**: Server 文件中忘记定义全局字典。
      * **检查**: 添加 `GLOBAL_SESSIONS = {}` 到文件顶部。