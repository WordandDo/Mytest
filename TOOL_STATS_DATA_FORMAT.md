# 工具调用统计 - 数据格式详解

## 📁 保存的数据格式

工具调用统计系统会保存两种格式的数据：

### 1. 实时日志文件 (realtime_calls.jsonl)

**位置**: `tool_stats/realtime_calls.jsonl`

**格式**: JSONL (JSON Lines) - 每行一个 JSON 对象

**特点**:
- ✅ **实时持久化**: 每 N 次调用（默认10次）自动追加到文件
- ✅ **进程安全**: 即使服务器崩溃，已记录的数据不会丢失
- ✅ **易于解析**: 逐行读取，内存占用小
- ✅ **可恢复**: 重启服务器时可以加载历史数据

**示例内容**:
```jsonl
{"tool_name": "search_documents", "task_id": "task_001", "timestamp": "2024-01-15T14:28:10.123456", "success": true, "error_message": null, "duration_ms": 234.56, "args": null}
{"tool_name": "read_file", "task_id": "task_001", "timestamp": "2024-01-15T14:28:15.345678", "success": true, "error_message": null, "duration_ms": 45.23, "args": null}
{"tool_name": "execute_bash", "task_id": "task_001", "timestamp": "2024-01-15T14:28:30.456789", "success": false, "error_message": "FileNotFoundError: /path/to/file not found", "duration_ms": 125.45, "args": {"args": [], "kwargs": {"command": "cat /path/to/file", "task_id": "task_001"}}}
{"tool_name": "search_documents", "task_id": "task_002", "timestamp": "2024-01-15T14:29:00.567890", "success": true, "error_message": null, "duration_ms": 312.78, "args": null}
```

**字段说明**:
- `tool_name` (string): 工具名称
- `task_id` (string): 任务 ID
- `timestamp` (string): ISO 8601 格式的时间戳
- `success` (boolean): 是否成功
- `error_message` (string|null): 错误信息（仅失败时有值）
- `duration_ms` (float|null): 执行耗时（毫秒）
- `args` (object|null): 调用参数（仅失败时记录）

### 2. 完整统计报告 (tool_stats_report_*.json)

**位置**: `tool_stats/tool_stats_report_YYYYMMDD_HHMMSS.json`

**格式**: JSON（美化格式，带缩进）

**生成时机**:
- 服务器正常关闭时（Ctrl+C 或 kill）
- 手动调用 `export_report()` 方法
- 使用 `view_tool_stats.py export` 命令

**完整结构** (见 [tool_stats_example.json](tool_stats_example.json)):
```json
{
  "generated_at": "2024-01-15T14:30:25.123456",
  "all_tasks_report": {
    "summary": { ... },
    "tasks": [ ... ]
  },
  "tool_report": {
    "total_tools": 3,
    "tools": [ ... ]
  },
  "failed_calls": [ ... ],
  "detailed_records": [ ... ]
}
```

## 🔄 数据持久化机制

### 运行时行为

```
主进程运行
    ↓
工具被调用
    ↓
记录到内存 (立即)
    ↓
每 N 次调用追加到 realtime_calls.jsonl (默认 N=10)
    ↓
继续运行...
    ↓
服务器关闭时
    ↓
生成完整报告 tool_stats_report_*.json
```

### 数据保留策略

| 场景 | 实时日志 (JSONL) | 完整报告 (JSON) | 内存数据 |
|------|------------------|-----------------|----------|
| **正常运行** | ✅ 持续追加 | ❌ 未生成 | ✅ 保留 |
| **进程崩溃** | ✅ 保留已写入的数据 | ❌ 丢失 | ❌ 丢失 |
| **正常关闭** | ✅ 保留所有数据 | ✅ 生成 | ✅ 可恢复 |
| **服务器重启** | ✅ 可加载 | ✅ 保留历史 | 🔄 可从 JSONL 恢复 |

### 实时保存配置

```python
# 默认配置（推荐）
collector = ToolStatsCollector(
    output_dir="tool_stats",
    enable_realtime_save=True,  # 启用实时保存
    save_interval=10             # 每10次调用保存一次
)

# 更频繁的保存（更安全但性能略降）
collector = ToolStatsCollector(
    output_dir="tool_stats",
    enable_realtime_save=True,
    save_interval=1  # 每次调用都保存
)

# 仅在关闭时保存（性能最好但不安全）
collector = ToolStatsCollector(
    output_dir="tool_stats",
    enable_realtime_save=False
)
```

## 💾 数据恢复示例

### 从实时日志恢复数据

```python
from mcp_server.core.tool_stats import ToolStatsCollector

# 创建收集器并加载历史数据
collector = ToolStatsCollector(output_dir="tool_stats")
loaded_count = collector.load_from_realtime_log()

print(f"Loaded {loaded_count} records from realtime log")

# 查看恢复的统计
collector.print_summary()
```

### 手动解析实时日志

```python
import json

with open('tool_stats/realtime_calls.jsonl', 'r') as f:
    for line in f:
        record = json.loads(line.strip())
        print(f"Task: {record['task_id']}, Tool: {record['tool_name']}, "
              f"Success: {record['success']}")
```

### 分析失败的调用

```bash
# 使用 jq 查询失败的调用
cat tool_stats/realtime_calls.jsonl | jq 'select(.success == false)'

# 统计每个工具的失败次数
cat tool_stats/realtime_calls.jsonl | jq -r 'select(.success == false) | .tool_name' | sort | uniq -c

# 查看特定任务的所有调用
cat tool_stats/realtime_calls.jsonl | jq 'select(.task_id == "task_001")'
```

## 📊 文件组织结构

```
tool_stats/
├── realtime_calls.jsonl                    # 实时日志（当前运行）
├── realtime_calls_backup_20240115_143000.jsonl  # 旧的实时日志备份
├── tool_stats_report_20240115_143000.json  # 完整报告 1
├── tool_stats_report_20240115_150000.json  # 完整报告 2
└── tool_stats_report_20240115_160000.json  # 完整报告 3
```

**说明**:
- `realtime_calls.jsonl`: 当前运行时的实时日志
- `realtime_calls_backup_*.jsonl`: 服务器重启时，旧的日志会被自动备份
- `tool_stats_report_*.json`: 每次正常关闭时生成，按时间戳命名

## 🔍 常见问题

### Q1: 数据会随主进程运行保留吗？

**A**: 是的！实时日志 (JSONL) 会在运行过程中持续保存。即使进程崩溃，已记录的数据也不会丢失。

### Q2: 服务器重启后数据会丢失吗？

**A**: 不会。实时日志文件会被保留，可以通过以下方式恢复：

```python
collector = ToolStatsCollector()
collector.load_from_realtime_log()  # 加载历史数据
```

或使用命令行工具：

```bash
python view_tool_stats.py summary  # 自动从实时日志加载数据
```

### Q3: 如何确保数据不丢失？

1. **使用实时保存**（默认已启用）
2. **降低保存间隔**（设置 `save_interval=1`）
3. **定期导出完整报告**

### Q4: 实时保存对性能有影响吗？

影响很小：
- 默认每10次调用才写入一次文件
- 使用追加模式（append），不需要重写整个文件
- 异步写入，不阻塞主线程
- 典型场景下，性能损失 < 1%

### Q5: 如何清理旧数据？

```bash
# 删除备份的实时日志
rm tool_stats/realtime_calls_backup_*.jsonl

# 删除旧的完整报告
rm tool_stats/tool_stats_report_*.json

# 清空当前实时日志（慎用！）
> tool_stats/realtime_calls.jsonl
```

## 🎯 最佳实践

1. **生产环境**:
   - 启用实时保存（`enable_realtime_save=True`）
   - 使用默认保存间隔（`save_interval=10`）
   - 定期备份 `tool_stats/` 目录

2. **开发/测试**:
   - 可以禁用实时保存以获得最佳性能
   - 关闭时会自动生成完整报告

3. **调试失败**:
   - 查看实时日志获取详细的失败记录
   - 使用 `view_tool_stats.py failures` 查看汇总

4. **长期监控**:
   - 定期导出完整报告并归档
   - 使用日志分析工具处理 JSONL 文件
