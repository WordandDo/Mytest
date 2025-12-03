# 分层超时策略设计

## 超时层级设计

### 第1层：LLM API调用超时 (API Level Timeout)
- **超时时间**: 30秒
- **作用范围**: 单次OpenAI API请求
- **触发行为**:
  - 抛出异常，进入重试逻辑
  - 最多重试2次（总计3次尝试）
  - 如果3次都超时，则任务失败
- **配置参数**:
  - `OPENAI_TIMEOUT=30`
  - `OPENAI_MAX_RETRIES=2`

### 第2层：任务执行超时 (Task Level Timeout)
- **超时时间**: 600秒（10分钟）
- **作用范围**: 从开始执行任务到完成任务的整个过程
- **触发行为**:
  - 记录超时日志
  - 强制释放资源
  - 标记任务为超时失败
  - 允许其他worker获取资源
- **配置参数**:
  - `TASK_EXECUTION_TIMEOUT=600`

### 第3层：资源占用超时 (Resource Occupation Timeout)
- **超时时间**: 900秒（15分钟，比任务超时更长）
- **作用范围**: 从资源分配到资源释放的整个周期
- **触发行为**:
  - 资源管理器层面强制回收资源
  - 记录异常日志
  - 清理残留状态
- **配置参数**:
  - `RESOURCE_MAX_OCCUPATION_TIME=900`

## 超时关系

```
API Timeout (30s)
    ↓ 重试3次 = 90s最大值
    ↓
Task Timeout (600s)
    ↓ 包含多轮API调用 + 工具执行
    ↓
Resource Timeout (900s)
    ↓ 安全边界，防止资源泄漏
```

## 实现要点

1. **不会误杀正常任务**
   - 任务超时600秒足够完成大部分任务
   - 如果任务需要更长时间，可以通过配置调整

2. **快速失败原则**
   - API层面30秒超时，避免长时间卡住
   - 3次重试后快速失败，释放资源

3. **优雅降级**
   - 超时时正确清理资源
   - 记录详细日志便于排查

4. **配置灵活性**
   - 所有超时时间可通过环境变量配置
   - 不同类型任务可以有不同的超时配置

## 配置示例

在 `.env` 文件中：
```bash
# API层超时（单次请求）
OPENAI_TIMEOUT=30
OPENAI_MAX_RETRIES=2

# 任务层超时（整个任务）
TASK_EXECUTION_TIMEOUT=600

# 资源层超时（安全边界）
RESOURCE_MAX_OCCUPATION_TIME=900
```

对于特别复杂的任务，可以调整：
```bash
TASK_EXECUTION_TIMEOUT=1800  # 30分钟
RESOURCE_MAX_OCCUPATION_TIME=2400  # 40分钟
```
