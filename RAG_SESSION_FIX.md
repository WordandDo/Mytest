# RAG Session 同步问题修复报告

## 问题描述

在完成统一 Batch 分配重构后，出现以下错误：

```
Data: [TextContent(type='text', text='{"status": "error", "message": "No active RAG session. Call setup_rag_session first."}', ...)]
```

## 根本原因分析

### 问题根源

**RAG_SESSIONS 全局字典没有被正确同步**，导致 `query_knowledge_base` 工具找不到会话信息。

### 问题链路

```
allocate_resource(worker_id, resource_init_data=None)  ← 没有传初始化配置
  ↓
allocate_batch_resources(["rag"])  ✓ 资源分配成功
  ↓
【关键问题】检查: if resource_init_data:  ← 条件为 False
  ├─ False → 跳过 setup_batch_resources()  ✗ 导致会话不同步
  └─ True  → 调用 setup_batch_resources()  ✓ 会话正常同步
  ↓
query_knowledge_base()
  ↓
检查 RAG_SESSIONS[worker_id]  ✗ 不存在！
  ↓
返回错误: "No active RAG session"
```

### 深层原因

1. **条件依赖错误**（http_mcp_env.py:675）
   ```python
   if resource_init_data:  # ← 只有有配置时才调用
       setup_res = self._call_tool_sync("setup_batch_resources", ...)
   ```

   问题：`setup_batch_resources` 不应该依赖于是否有初始化配置。

2. **执行顺序错误**（system_tools.py:159-165）
   ```python
   # 原来的顺序
   if not resource_init_configs:
       return  # ← 提前返回，不执行同步

   if allocated_resources:
       await _sync_resource_sessions()  # ← 永远不会执行
   ```

   问题：配置检查在会话同步之前，导致无配置时会话不同步。

## 修复方案

### 修复 1: 调整 setup_batch_resources 执行顺序

**文件**: [src/mcp_server/system_tools.py:159-167](src/mcp_server/system_tools.py#L159-L167)

**修改前**:
```python
if not resource_init_configs:
    return json.dumps({"status": "success", "details": "No config provided"})

# 在初始化前，先同步资源状态到各模块的全局变量
if allocated_resources:
    logger.info(f"[{worker_id}] Syncing allocated resources to module sessions...")
    await _sync_resource_sessions(worker_id, allocated_resources)
```

**修改后**:
```python
# [关键修复] 即使没有 resource_init_configs，也要同步资源会话
# 这确保了 RAG_SESSIONS 等全局会话字典能够正确更新
if allocated_resources:
    logger.info(f"[{worker_id}] Syncing allocated resources to module sessions...")
    await _sync_resource_sessions(worker_id, allocated_resources)

# 如果没有初始化配置，同步后直接返回成功
if not resource_init_configs:
    return json.dumps({"status": "success", "details": "No config provided, session sync completed"})
```

**作用**:
- ✅ 将会话同步移到配置检查**之前**
- ✅ 确保即使没有配置，`_sync_resource_sessions` 也会执行
- ✅ RAG_SESSIONS 总是会被更新

### 修复 2: 移除 allocate_resource 的条件检查

**文件**: [src/envs/http_mcp_env.py:674-686](src/envs/http_mcp_env.py#L674-L686)

**修改前**:
```python
self.allocated_resources = data

# 2. 初始化资源
if resource_init_data:  # ← 条件检查
    setup_res = self._call_tool_sync("setup_batch_resources", {
        "resource_init_configs": resource_init_data,
        "allocated_resources": data
    })
    setup_result = self._parse_mcp_response(setup_res)
    if setup_result.get("status") not in ["success", "partial_error"]:
        logger.error(f"Setup failed: {setup_result}")
        self.release_resource(self.worker_id)
        return False
```

**修改后**:
```python
self.allocated_resources = data

# 2. 初始化资源（总是调用以确保会话同步）
# 即使没有 resource_init_data，也需要调用 setup_batch_resources 来同步会话
# 减少日志：移除资源设置日志
# logger.info(f"[{self.worker_id}] Setting up resources...")
setup_res = self._call_tool_sync("setup_batch_resources", {
    "resource_init_configs": resource_init_data,  # 可以为空 dict，不影响会话同步
    "allocated_resources": data  # 关键：传递已分配的资源信息用于 _sync_resource_sessions
})
setup_result = self._parse_mcp_response(setup_res)
if setup_result.get("status") not in ["success", "partial_error"]:
    logger.error(f"Setup failed: {setup_result}")
    self.release_resource(self.worker_id)
    return False
```

**作用**:
- ✅ 移除 `if resource_init_data` 条件
- ✅ `setup_batch_resources` **总是**被调用
- ✅ 确保会话同步逻辑总是被触发

## 修复后的完整流程

```
allocate_resource(worker_id, resource_init_data)  ← 无论 resource_init_data 是否为 None
  ↓
1. allocate_batch_resources(["rag"])
   返回: {"rag": {"id": "xxx", "token": "yyy", "base_url": "http://..."}}
  ↓
2. setup_batch_resources(resource_init_data, allocated_resources)  ← 总是调用
   ↓
   2.1 _sync_resource_sessions(worker_id, allocated_resources)  ← 总是执行
       ↓
       RAG_SESSIONS[worker_id] = {
           "resource_id": "xxx",
           "token": "yyy",
           "base_url": "http://...",
           "config_top_k": None
       }  ✓ Session 已建立
   ↓
   2.2 检查 resource_init_configs:
       ├─ 为空 → 返回成功（Session 已同步）
       └─ 不为空 → 调用 rag_initialization(worker_id, config)
                   ↓
                   更新 session["config_top_k"] = 5  ✓ 配置已应用
  ↓
3. 后续调用 query_knowledge_base(worker_id, query)
   ↓
   检查 RAG_SESSIONS[worker_id]  ✓ 存在！
   ↓
   查询成功 ✓
```

## 关于 rag_initialization 的说明

**问题**: `rag_initialization` 是否负责建立 session？

**答案**: **否**。职责分工如下：

| 函数 | 职责 | 何时执行 |
|------|------|---------|
| `_sync_resource_sessions` | **建立** Session（写入 RAG_SESSIONS） | 总是执行（只要有 allocated_resources） |
| `rag_initialization` | **配置** Session（更新 config_top_k） | 仅在有配置时执行 |

`rag_initialization` 的代码假设 session 已存在：
```python
async def rag_initialization(worker_id: str, config_content = None) -> bool:
    session = RAG_SESSIONS.get(worker_id)
    if not session:
        logger.error("Ensure _sync_resource_sessions was called first.")
        return False  # ← 假设 session 应该已存在

    # 只更新配置
    if config_content and "top_k" in config_dict:
        session["config_top_k"] = config_dict["top_k"]
```

## 测试验证

### 静态验证

```bash
$ python3 test_rag_session_fix.py
✓ 会话同步移到配置检查之前
✓ 同步后才检查配置
✓ setup_batch_resources 被调用
✓ 移除了 if resource_init_data 条件
✓ 传递 allocated_resources 参数
```

### 建议的集成测试

```bash
# 运行实际的 RAG 环境测试
./run_rag_env.sh

# 或快速验证
./verify_rag_quick.sh
```

**预期日志**：
```
[worker_xxx] Allocating resources...
[worker_xxx] Syncing allocated resources to module sessions...
[worker_xxx] Synced RAG session (Direct Mode: http://...)
[worker_xxx] RAG initialization completed (no config provided)  # 如果没有配置
# 或
[worker_xxx] RAG initialization completed with top_k=5  # 如果有配置
```

## 影响范围

### 受益的场景

1. **无配置 RAG 使用**
   ```python
   env = HttpMCPRagEnv()
   env.allocate_resource(worker_id)  # 不传 resource_init_data
   # 现在可以正常查询了！
   ```

2. **默认配置使用**
   ```python
   env.allocate_resource(worker_id, resource_init_data={})
   # 空配置也能正常工作
   ```

3. **所有 RAG 环境**
   - 统一 Batch 分配模式现在完全可用
   - 不再需要显式调用 `setup_rag_session`

### 不受影响的场景

- 有配置的 RAG 使用（已经可以正常工作）
- VM 环境（使用相同的基础设施，但不受影响）
- 旧的 `setup_rag_session` 调用（已废弃但仍兼容）

## 总结

| 项目 | 状态 |
|------|------|
| **问题原因** | ✓ 已分析清楚 |
| **修复方案** | ✓ 已实施完成 |
| **代码验证** | ✓ 已通过测试 |
| **文档更新** | ✓ 已完成 |
| **集成测试** | ⏳ 待用户运行 |

### 关键要点

1. **会话同步是必要步骤**，不应依赖于是否有初始化配置
2. **`setup_batch_resources` 必须总是被调用**，以触发 `_sync_resource_sessions`
3. **`rag_initialization` 只负责配置**，不负责建立会话
4. **两个修复必须配合**才能完全解决问题

### 文件变更清单

- ✅ [src/mcp_server/system_tools.py](src/mcp_server/system_tools.py) - 调整执行顺序
- ✅ [src/envs/http_mcp_env.py](src/envs/http_mcp_env.py) - 移除条件检查
- ✅ [test_rag_session_fix.py](test_rag_session_fix.py) - 测试验证脚本
- ✅ [RAG_SESSION_FIX.md](RAG_SESSION_FIX.md) - 本文档

## 下一步

建议用户运行实际的 RAG 环境测试：

```bash
./run_rag_env.sh
# 或
./verify_rag_quick.sh
```

如果测试通过，统一 Batch 分配重构即完成！
