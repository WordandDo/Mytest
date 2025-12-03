# 超时配置参数说明

## 环境变量配置

在`.env`文件中添加以下配置（所有参数都是可选的，有默认值）：

```bash
# ============================================================
# 第1层超时：OpenAI API调用超时配置
# ============================================================
# 单次API请求的最大等待时间（秒）
# 建议值：30-60秒
# 默认值：30秒
OPENAI_TIMEOUT=30

# API请求失败后的最大重试次数
# 建议值：2-3次
# 默认值：2次
OPENAI_MAX_RETRIES=2

# ============================================================
# 第2层超时：任务执行超时配置
# ============================================================
# 单个任务的最大执行时间（秒）
# 包含多轮对话、工具调用等所有操作
# 建议值：300-1800秒（5-30分钟）
# 默认值：600秒（10分钟）
TASK_EXECUTION_TIMEOUT=600

# ============================================================
# 第3层超时：资源占用超时配置
# ============================================================
# 资源被单个worker占用的最大时间（秒）
# 这是安全边界，防止资源泄漏
# 建议值：比TASK_EXECUTION_TIMEOUT大50%
# 默认值：900秒（15分钟）
RESOURCE_MAX_OCCUPATION_TIME=900
```

## 超时层级关系

```
API Timeout (30s)              # 快速失败，避免长时间阻塞
    ↓
    × max_retries (2)
    ↓
= 最多90秒的API调用时间

Task Timeout (600s)            # 整个任务的时间限制
    ↓
    包含多轮对话
    ↓
    包含工具执行
    ↓
= 10分钟内必须完成任务

Resource Timeout (900s)        # 安全边界，强制回收
    ↓
    防止资源泄漏
    ↓
    强制释放并记录异常
```

## 使用示例

### 场景1：快速测试（缩短超时）
```bash
# 适用于开发和调试
OPENAI_TIMEOUT=10
OPENAI_MAX_RETRIES=1
TASK_EXECUTION_TIMEOUT=60
RESOURCE_MAX_OCCUPATION_TIME=90
```

### 场景2：正常运行（默认配置）
```bash
# 适用于生产环境
OPENAI_TIMEOUT=30
OPENAI_MAX_RETRIES=2
TASK_EXECUTION_TIMEOUT=600
RESOURCE_MAX_OCCUPATION_TIME=900
```

### 场景3：复杂任务（延长超时）
```bash
# 适用于需要长时间运行的复杂任务
OPENAI_TIMEOUT=60
OPENAI_MAX_RETRIES=3
TASK_EXECUTION_TIMEOUT=1800
RESOURCE_MAX_OCCUPATION_TIME=2400
```

## 超时行为说明

### 第1层：API超时
- **触发条件**：OpenAI API单次请求超过OPENAI_TIMEOUT秒
- **行为**：
  1. 抛出超时异常
  2. 自动重试（最多OPENAI_MAX_RETRIES次）
  3. 所有重试失败后，任务失败
- **日志标识**：`[APITimeout]`
- **不会释放资源**：重试期间资源保持占用

### 第2层：任务超时
- **触发条件**：任务执行时间超过TASK_EXECUTION_TIMEOUT秒
- **行为**：
  1. 抛出TaskTimeoutError异常
  2. 停止任务执行
  3. 立即释放资源
  4. 记录任务失败
- **日志标识**：`⏰ [TaskTimeout]`
- **资源处理**：正常释放（调用release方法）

### 第3层：资源强制回收
- **触发条件**：资源被占用超过RESOURCE_MAX_OCCUPATION_TIME秒
- **行为**：
  1. 资源监控器自动检测
  2. 强制释放资源
  3. 记录严重警告
  4. 资源状态重置
- **日志标识**：`🚨 [ResourceTimeout]`
- **检查频率**：每30秒检查一次（在monitor_resource_usage中）

## 监控和日志

### 关键日志示例

#### API超时
```
WARNING - Retry 1/2 due to error: TimeoutError: Request timeout after 30s
```

#### 任务超时
```
ERROR - ⏰ [TaskTimeout] Worker=worker-1 Task=task-001
timeout check failed: 605.3s > 600s
ERROR - ❌ [TaskTimeout] Task task-001 timeout:
Task timeout after 605.3s (limit: 600s) at turn 2
```

#### 资源强制回收
```
ERROR - 🚨 [ResourceTimeout] Force reclaiming vm_1 from worker-1
after 920.5s (limit: 900s)
INFO - ♻️ [ForcedRelease] vm_1 reclaimed
(was occupied by worker-1 for 920.5s)
```

## 故障排查

### 问题1：任务经常超时
**症状**：大量`[TaskTimeout]`日志
**解决**：
1. 增加`TASK_EXECUTION_TIMEOUT`
2. 检查网络连接
3. 优化任务复杂度

### 问题2：资源被强制回收
**症状**：出现`[ResourceTimeout]`日志
**解决**：
1. 检查是否有worker卡住
2. 增加`RESOURCE_MAX_OCCUPATION_TIME`
3. 检查任务是否正常完成

### 问题3：API频繁超时
**症状**：大量API重试日志
**解决**：
1. 检查API endpoint可用性
2. 增加`OPENAI_TIMEOUT`
3. 检查网络连接质量
4. 验证API Key是否有效

## 性能影响

### CPU和内存
- 超时监控几乎无性能开销
- 使用线程Timer，不阻塞主线程

### 响应时间
- API超时：可能增加(重试次数 × 超时时间)的延迟
- 任务超时：立即停止，释放资源
- 资源超时：每30秒检查一次，开销可忽略

## 注意事项

1. **不要设置过短的超时时间**
   - API timeout < 10秒：可能导致正常请求失败
   - Task timeout < 60秒：可能无法完成简单任务

2. **保持超时层级关系**
   - `RESOURCE_MAX_OCCUPATION_TIME > TASK_EXECUTION_TIMEOUT`
   - `TASK_EXECUTION_TIMEOUT > (OPENAI_TIMEOUT × OPENAI_MAX_RETRIES × max_turns)`

3. **监控资源回收**
   - 如果频繁出现第3层超时，说明系统有问题
   - 正常情况下应该很少触发资源强制回收
