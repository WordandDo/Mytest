# 超时检测快速使用指南

## 1. 快速启动

### 步骤1：添加环境变量（可选）
在`.env`文件中添加：
```bash
# 使用默认值（推荐）
OPENAI_TIMEOUT=30
TASK_EXECUTION_TIMEOUT=600
RESOURCE_MAX_OCCUPATION_TIME=900

# 或者根据需求调整
```

### 步骤2：直接运行
```bash
# 系统会自动使用默认的超时配置
python src/run_parallel_rollout.py --config your_config.json
```

**就这么简单！无需任何代码修改，超时检测已经自动工作。**

## 2. 验证功能是否生效

### 查看启动日志
```bash
# 应该看到类似输出：
[worker-1] Initializing OpenAI client with timeout=30.0s, max_retries=2
⏱️  [TaskTimeout] Started monitoring for Worker=worker-1 Task=task-001, timeout=600s
VMPoolImpl initialized with 1 items, max_occupation_time=900.0s
```

### 观察监控日志
```bash
# 每30秒会输出资源状态：
📊 [Monitor] VM_PYAUTOGUI(Free:0/1) RAG(Free:2/3)
```

### 检查超时日志
```bash
# 如果任务超时，会看到：
⏰ [TaskTimeout] Worker=worker-1 Task=task-001 timeout check failed
❌ [TaskTimeout] Task task-001 timeout: Task timeout after 605.3s

# 如果资源被强制回收，会看到：
🚨 [ResourceTimeout] Force reclaiming vm_1 from worker-1
♻️ [ForcedRelease] vm_1 reclaimed
```

## 3. 常见使用场景

### 场景1：解决原问题（Worker卡住占用资源）

**问题描述**：Worker-1因API超时卡住10分钟，其他worker等待超时

**解决方案**：已自动生效！
- API超时30秒后自动重试
- 任务超时600秒后自动释放资源
- 资源强制回收900秒作为最后保障

**验证方法**：
```bash
# 运行并观察日志
tail -f logs/resource_api.log | grep -E "Timeout|Released"
```

### 场景2：快速测试（缩短超时）

**需求**：开发调试时希望快速失败

**配置**：
```bash
export OPENAI_TIMEOUT=10
export TASK_EXECUTION_TIMEOUT=60
python src/run_parallel_rollout.py ...
```

### 场景3：复杂任务（延长超时）

**需求**：某些任务确实需要很长时间

**配置**：
```bash
export TASK_EXECUTION_TIMEOUT=1800  # 30分钟
export RESOURCE_MAX_OCCUPATION_TIME=2400  # 40分钟
python src/run_parallel_rollout.py ...
```

## 4. 常见问题

### Q1：我的任务会被误杀吗？
**A**：不会！超时时间设置合理：
- API超时：30秒（单次请求）
- 任务超时：600秒（整个任务）
- 资源超时：900秒（安全边界）

正常任务在600秒内完成不会被终止。

### Q2：如何知道任务超时了？
**A**：查看日志和结果文件：
```bash
# 日志中会有 TaskTimeout 标识
grep "TaskTimeout" logs/client_run.log

# 结果文件中 error 字段会显示超时
cat results/*/trajectory.jsonl | jq '.error'
```

### Q3：超时后资源真的会释放吗？
**A**：是的！通过三层保障：
1. TaskTimeoutError异常 → run_task的finally块释放
2. Worker的finally块 → 确保资源释放
3. 监控器每30秒检查 → 强制回收泄漏资源

### Q4：需要修改代码吗？
**A**：不需要！所有功能已集成到框架中，只需配置环境变量。

### Q5：如何调整超时时间？
**A**：两种方式：
```bash
# 方式1：环境变量（推荐）
export OPENAI_TIMEOUT=60

# 方式2：.env文件
echo "OPENAI_TIMEOUT=60" >> .env
```

## 5. 监控和调试

### 实时监控资源状态
```bash
# 查看资源池状态
watch -n 5 'tail -20 logs/resource_api.log | grep Monitor'
```

### 查看超时事件
```bash
# 查看所有超时日志
grep -E "Timeout|TaskTimeout|ResourceTimeout" logs/*.log

# 统计超时次数
grep "TaskTimeout" logs/*.log | wc -l
```

### 分析任务耗时
```bash
# 查看任务完成时间
grep "FINISH Task" logs/client_run.log
```

## 6. 性能影响

**CPU**：几乎无影响（<0.1%）
**内存**：每个任务约1KB额外开销
**延迟**：正常情况无影响，超时时快速失败

## 7. 推荐配置

### 生产环境（默认）
```bash
OPENAI_TIMEOUT=30
OPENAI_MAX_RETRIES=2
TASK_EXECUTION_TIMEOUT=600
RESOURCE_MAX_OCCUPATION_TIME=900
```

### 开发环境
```bash
OPENAI_TIMEOUT=15
OPENAI_MAX_RETRIES=1
TASK_EXECUTION_TIMEOUT=300
RESOURCE_MAX_OCCUPATION_TIME=450
```

### 长时间任务
```bash
OPENAI_TIMEOUT=60
OPENAI_MAX_RETRIES=3
TASK_EXECUTION_TIMEOUT=1800
RESOURCE_MAX_OCCUPATION_TIME=2400
```

## 8. 故障排查

### 问题：任务经常超时
```bash
# 1. 检查网络连接
curl -I http://your-api-endpoint

# 2. 验证API Key
echo $OPENAI_API_KEY

# 3. 增加超时时间
export TASK_EXECUTION_TIMEOUT=1200
```

### 问题：看不到超时日志
```bash
# 1. 确认日志级别
export LOG_LEVEL=INFO

# 2. 检查日志文件
ls -lh logs/

# 3. 手动触发超时测试
export TASK_EXECUTION_TIMEOUT=10  # 极短超时
```

## 9. 获取帮助

### 查看文档
- **设计文档**：[docs/timeout_strategy.md](docs/timeout_strategy.md)
- **配置说明**：[docs/timeout_configuration.md](docs/timeout_configuration.md)
- **实现总结**：[docs/timeout_implementation_summary.md](docs/timeout_implementation_summary.md)

### 查看代码
- **监控工具**：[src/utils/task_timeout.py](src/utils/task_timeout.py)
- **环境配置**：[src/envs/http_mcp_env.py](src/envs/http_mcp_env.py)
- **资源池**：[src/utils/resource_pools/base.py](src/utils/resource_pools/base.py)

## 10. 最后

**记住**：超时检测是为了保护资源，不是限制任务。合理配置超时时间，系统会自动处理异常情况，确保资源高效利用。

**默认配置已经很合理**，大多数情况下无需调整！
