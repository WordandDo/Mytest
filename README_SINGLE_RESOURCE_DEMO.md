# 单资源分配与释放流程演示

这个演示展示了如何使用 `src/mcp_server/system_tools.py` 中的单资源分配工具实现对单个资源的完整生命周期管理。

## 架构概览

```
┌─────────────────────────────────────────────────────────┐
│         demo_single_resource_flow.py                    │
│         (SimpleSingleResourceDemo)                      │
└─────────────────┬───────────────────────────────────────┘
                  │ MCP SSE Protocol
                  ▼
┌─────────────────────────────────────────────────────────┐
│         MCP Gateway Server                              │
│         (localhost:8080)                                │
│  ┌──────────────────────────────────────────────────┐  │
│  │  system_tools.py                                 │  │
│  │  - allocate_single_resource()                    │  │
│  │  - setup_batch_resources()                       │  │
│  │  - get_batch_initial_observations()              │  │
│  │  - release_batch_resources()                     │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────┬───────────────────────────────────────┘
                  │ HTTP API
                  ▼
┌─────────────────────────────────────────────────────────┐
│         Resource Manager API                            │
│         (localhost:8000)                                │
│  - /allocate (资源分配)                                 │
│  - /release (资源释放)                                  │
└─────────────────────────────────────────────────────────┘
```

## 核心流程

演示程序实现了以下完整流程：

### 1. 初始化阶段
```python
# 创建演示实例
demo = SimpleSingleResourceDemo(server_url="http://localhost:8080")

# 连接到 MCP Server
await demo.connect()

# 列出可用工具
tools = await demo.list_available_tools()
```

### 2. 资源分配阶段
```python
# 使用 allocate_single_resource 工具分配单个资源
allocation_success = await demo.allocate_single_resource("vm_pyautogui")

# 返回格式示例：
# {
#   "vm_pyautogui": {
#     "id": "resource_001",
#     "ip": "192.168.1.100",
#     "port": 5000,
#     "token": "auth_token_123"
#   }
# }
```

### 3. 资源初始化阶段
```python
# 使用 setup_batch_resources 工具初始化资源
# 这会同步资源会话到相应的 Server 模块（如 vm_pyautogui_server）
setup_success = await demo.setup_resource(allocated_data, init_config={})
```

### 4. 获取初始观察
```python
# 使用 get_batch_initial_observations 工具获取初始状态
observation = await demo.get_initial_observation()

# 返回格式示例：
# {
#   "vm_pyautogui": {
#     "screenshot": "base64_encoded_image...",
#     "accessibility_tree": "tree_data...",
#     "message": "Observation fetched from local controller"
#   }
# }
```

### 5. 工具调用阶段
```python
# 调用资源相关的工具（示例）
await demo.call_tool_example("computer", {"action": "screenshot"})
```

### 6. 资源释放阶段
```python
# 使用 release_batch_resources 工具释放资源
await demo.release_resource()

# 这会：
# 1. 向 Resource Manager 发送释放请求
# 2. 清理 Gateway 侧的全局会话缓存
```

## 使用方法

### 前置要求

1. **Resource Manager API** 运行在 `localhost:8000`
2. **MCP Gateway Server** 运行在 `localhost:8080`
3. 已配置的资源池（包含 vm_pyautogui 或其他资源类型）

### 运行演示

#### 基本运行
```bash
python demo_single_resource_flow.py
```

#### 自定义配置
```bash
# 指定 MCP Server 地址
MCP_SERVER_URL=http://localhost:8080 python demo_single_resource_flow.py

# 指定资源类型
DEMO_RESOURCE_TYPE=vm_computer_13 python demo_single_resource_flow.py

# 指定 Resource API 地址
RESOURCE_API_URL=http://localhost:8000 python demo_single_resource_flow.py

# 组合使用
MCP_SERVER_URL=http://localhost:8080 \
DEMO_RESOURCE_TYPE=vm_computer_13 \
RESOURCE_API_URL=http://localhost:8000 \
python demo_single_resource_flow.py
```

## 支持的资源类型

根据 `system_tools.py` 的实现，当前支持以下资源类型：

- `vm_pyautogui`: PyAutoGUI 桌面控制环境
- `vm_computer_13`: Computer Use 桌面控制环境
- `rag`: RAG 检索服务
- `rag_hybrid`: 混合 RAG 检索服务

## 输出示例

```
2025-12-10 10:00:00 - __main__ - INFO - [demo_worker_001] SimpleSingleResourceDemo initialized
2025-12-10 10:00:00 - __main__ - INFO - [demo_worker_001] Server URL: http://localhost:8080
2025-12-10 10:00:01 - __main__ - INFO - [demo_worker_001] Connecting to MCP Server...
2025-12-10 10:00:01 - __main__ - INFO - [demo_worker_001] ✅ Connected to MCP Server
2025-12-10 10:00:01 - __main__ - INFO - [demo_worker_001] Fetching available tools...
2025-12-10 10:00:02 - __main__ - INFO - [demo_worker_001] Found 15 tools:
2025-12-10 10:00:02 - __main__ - INFO -   - allocate_single_resource: [System Tool] Allocate a single resource...
2025-12-10 10:00:02 - __main__ - INFO -   - release_batch_resources: [System Tool] Batch release resources...
2025-12-10 10:00:02 - __main__ - INFO - [demo_worker_001] Allocating single resource: vm_pyautogui
2025-12-10 10:00:05 - __main__ - INFO - [demo_worker_001] ✅ Resource allocated successfully
2025-12-10 10:00:05 - __main__ - INFO - [demo_worker_001]   Resource ID: resource_001
2025-12-10 10:00:05 - __main__ - INFO - [demo_worker_001] Setting up resource...
2025-12-10 10:00:06 - __main__ - INFO - [demo_worker_001] ✅ Resource setup completed
2025-12-10 10:00:06 - __main__ - INFO - [demo_worker_001] Fetching initial observation...
2025-12-10 10:00:07 - __main__ - INFO - [demo_worker_001] ✅ Initial observation retrieved
2025-12-10 10:00:07 - __main__ - INFO - [demo_worker_001]   vm_pyautogui:
2025-12-10 10:00:07 - __main__ - INFO - [demo_worker_001]     - Screenshot: Yes
2025-12-10 10:00:07 - __main__ - INFO - [demo_worker_001]     - Accessibility Tree: Yes
2025-12-10 10:00:07 - __main__ - INFO - [demo_worker_001] Releasing resource: resource_001
2025-12-10 10:00:08 - __main__ - INFO - [demo_worker_001] ✅ Resource released successfully
2025-12-10 10:00:08 - __main__ - INFO - ✅ Demo completed successfully!
```

## 与现有环境的对比

### HttpMCPEnv (批量资源分配)
- 使用 `allocate_batch_resources` 一次性分配多个资源
- 适合需要多种资源协同工作的场景（如 VM + RAG）
- 示例：[src/envs/http_mcp_env.py:775-778](src/envs/http_mcp_env.py#L775-L778)

### HttpMCPRagEnv (单一资源类型)
- 继承自 HttpMCPEnv
- 仅分配 RAG 类型资源
- 使用批量接口但只传递单一资源类型
- 示例：[src/envs/http_mcp_rag_env.py:164-195](src/envs/http_mcp_rag_env.py#L164-L195)

### HttpMCPSearchEnv (无状态工具)
- 继承自 HttpMCPEnv
- 不需要资源分配（`active_resources = []`）
- 仅使用无状态工具（如 web_search）
- 示例：[src/envs/http_mcp_search_env.py:56-58](src/envs/http_mcp_search_env.py#L56-L58)

### SimpleSingleResourceDemo (本演示)
- **独立演示**，不继承 HttpMCPEnv
- 直接使用 `allocate_single_resource` 工具
- 展示单资源分配的最简化流程
- 适合学习和理解资源管理机制

## 关键工具说明

### allocate_single_resource
- **位置**: [src/mcp_server/system_tools.py:47-85](src/mcp_server/system_tools.py#L47-L85)
- **功能**: 分配单个指定类型的资源
- **参数**:
  - `worker_id`: Worker ID
  - `resource_type`: 资源类型
  - `timeout`: 超时时间（默认 600 秒）
- **返回**: 资源信息（包含 id, ip, port, token 等）

### setup_batch_resources
- **位置**: [src/mcp_server/system_tools.py:185-260](src/mcp_server/system_tools.py#L185-L260)
- **功能**: 初始化资源并同步会话
- **关键**: 即使没有 init_config，也会调用 `_sync_resource_sessions` 同步会话

### get_batch_initial_observations
- **位置**: [src/mcp_server/system_tools.py:121-182](src/mcp_server/system_tools.py#L121-L182)
- **功能**: 从本地 Controller 获取初始观察数据
- **支持**: 截图、可访问性树等

### release_batch_resources
- **位置**: [src/mcp_server/system_tools.py:88-118](src/mcp_server/system_tools.py#L88-L118)
- **功能**: 批量释放资源（支持单资源）
- **清理**: 自动调用 `_cleanup_resource_sessions` 清理会话缓存

## 扩展示例

如果你想在演示中添加具体的工具调用，可以参考：

### VM 控制工具调用
```python
# 获取屏幕截图
await demo.call_tool_example(
    "computer",
    {
        "action": "screenshot"
    }
)

# 移动鼠标
await demo.call_tool_example(
    "computer",
    {
        "action": "mouse_move",
        "coordinate": [100, 200]
    }
)
```

### RAG 查询工具调用
```python
# 执行检索查询
await demo.call_tool_example(
    "rag_query",
    {
        "query": "What is the capital of France?",
        "top_k": 5
    }
)
```

## 故障排除

### 连接失败
```
Error: Failed to connect to MCP Server
```
**解决方案**: 确保 MCP Gateway Server 运行在指定端口

### 资源分配失败
```
Error: System Single Allocation Failed for vm_pyautogui
```
**可能原因**:
1. Resource Manager API 未运行
2. 资源池中没有可用资源
3. 资源类型不存在

### 工具调用失败
```
Error: Tool call failed
```
**检查**:
1. 资源是否已正确分配和初始化
2. worker_id 是否正确传递
3. 工具参数是否符合要求

## 参考资料

- [HttpMCPEnv 完整实现](src/envs/http_mcp_env.py)
- [System Tools 实现](src/mcp_server/system_tools.py)
- [MCP SSE Client](utils/mcp_sse_client.py)
